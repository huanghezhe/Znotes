import re
from pathlib import Path
from typing import List, Dict

from flask import Flask, render_template, request, abort, send_from_directory, jsonify
try:
    import jieba
    _HAS_JIEBA = True
except Exception:
    _HAS_JIEBA = False
import markdown2
import math
from collections import Counter

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
NOTES_DIR = ROOT / "notes"
TEMPLATE_DIR = ROOT / "templates"
STATIC_DIR = ROOT / "static"

app.template_folder = str(TEMPLATE_DIR)
app.static_folder = str(STATIC_DIR)


def iter_markdown_files(base: Path) -> List[Path]:
    files = []
    for path in sorted(base.rglob("*.md")):
        if path.is_file():
            files.append(path)
    return files


def build_tree(base: Path) -> List[Dict[str, object]]:
    root = {"type": "folder", "name": "", "path": "", "children": []}
    for path in sorted(base.rglob("*.md")):
        if not path.is_file():
            continue
        rel = path.relative_to(base)
        current = root
        for part in rel.parts[:-1]:
            child = next((item for item in current["children"] if item["type"] == "folder" and item["name"] == part), None)
            if child is None:
                child_path = f"{current['path']}/{part}" if current["path"] else part
                child = {"type": "folder", "name": part, "path": child_path, "children": []}
                current["children"].append(child)
            current = child
        file_path = "/".join(rel.parts)
        current["children"].append({
            "type": "file",
            "name": Path(rel.name).stem,
            "path": file_path,
            "title": Path(rel.name).stem,
        })

    def sort_nodes(nodes: List[Dict[str, object]]) -> List[Dict[str, object]]:
        nodes.sort(key=lambda item: (item["type"] != "folder", str(item["name"]).lower()))
        for node in nodes:
            if node["type"] == "folder" and node.get("children"):
                node["children"] = sort_nodes(node["children"])
        return nodes

    return sort_nodes(root["children"])


def read_note(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class SearchIndex:
    def __init__(self, base: Path):
        self.base = base
        self.docs = []  # list of dicts: {path,str,title,content}
        self.vocab = set()
        self.df = Counter()
        self.idf = {}
        self.doc_tfs = []
        self.doc_vecs = []
        self._build()

    def _tokenize(self, text: str):
        if not text:
            return []
        # if contains CJK characters, prefer jieba segmentation
        if re.search(r"[\u4e00-\u9fff]", text):
            if _HAS_JIEBA:
                segs = [t for t in jieba.cut(text) if t.strip()]
                # add ascii tokens as well
                ascii_tokens = re.findall(r"\w+", text.lower())
                return [s.lower() for s in segs if s.strip()] + ascii_tokens
            else:
                # fallback: split CJK to characters plus ascii tokens
                chars = re.findall(r"[\u4e00-\u9fff]", text)
                ascii_tokens = re.findall(r"\w+", text.lower())
                return chars + ascii_tokens
        # default: ascii tokenization
        return re.findall(r"\w+", text.lower())

    def _build(self):
        paths = sorted(self.base.rglob("*.md"))
        for p in paths:
            if not p.is_file():
                continue
            content = read_note(p)
            tokens = self._tokenize(content)
            tf = Counter(tokens)
            relpath = p.relative_to(self.base).as_posix()
            title = p.stem
            # tokenize title and path for name-based matching
            title_tokens = self._tokenize(title)
            path_tokens = []
            for part in relpath.split('/'):
                path_tokens.extend(self._tokenize(part))
            self.docs.append({"path": relpath, "title": title, "content": content, "tokens": tokens, "title_tokens": title_tokens, "path_tokens": path_tokens})
            self.doc_tfs.append(tf)
            for t in set(tokens):
                self.df[t] += 1
            self.vocab.update(tokens)

        # document lengths
        self.doc_lens = [len(d['tokens']) for d in self.docs]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        # compute BM25-style idf
        N = max(1, len(self.docs))
        for term, df in self.df.items():
            # BM25 idf variant
            self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1e-9)

        # store raw doc vectors (tf) for BM25 scoring
        self.doc_vecs = [dict(tf) for tf in self.doc_tfs]

    def reindex(self):
        # rebuild from scratch
        self.docs = []
        self.vocab = set()
        self.df = Counter()
        self.idf = {}
        self.doc_tfs = []
        self.doc_vecs = []
        self._build()

    def search(self, query: str, top_n: int = 10):
        qtokens = self._tokenize(query)
        if not qtokens:
            return []

        # BM25 parameters
        k1 = 1.5
        b = 0.75
        title_boost = 1.6
        path_boost = 1.2

        q_terms = Counter(qtokens)
        scores = []
        for i, doc in enumerate(self.docs):
            dl = self.doc_lens[i]
            score = 0.0
            doc_tf = self.doc_vecs[i]
            for term, qf in q_terms.items():
                idf = self.idf.get(term, math.log((len(self.docs) + 1) / 1) )
                f = doc_tf.get(term, 0)
                if f > 0:
                    denom = f + k1 * (1 - b + b * dl / max(1.0, self.avgdl))
                    score += idf * (f * (k1 + 1)) / denom

            # title boost: count term occurrences in title
            # title and path token matches (use tokens to avoid substring artifacts)
            title_tokens = doc.get('title_tokens', [])
            path_tokens = doc.get('path_tokens', [])
            title_matches = 0
            path_matches = 0
            for t in q_terms:
                title_matches += title_tokens.count(t)
                path_matches += path_tokens.count(t)
            if title_matches:
                # weight by idf of matched terms
                for t in set(q_terms):
                    if t in title_tokens:
                        score += title_boost * self.idf.get(t, 1.0)
            if path_matches:
                for t in set(q_terms):
                    if t in path_tokens:
                        score += path_boost * self.idf.get(t, 0.5)

            # include documents that match title/path even if content score is zero
            if score > 0 or title_matches or path_matches:
                scores.append((score, i))

        scores.sort(reverse=True)
        results = []
        for score, idx in scores[:top_n]:
            doc = self.docs[idx]
            snippet = self._make_snippet(doc['content'], qtokens)
            # determine match origins
            doc_title_tokens = set(doc.get('title_tokens', []))
            doc_path_tokens = set(doc.get('path_tokens', []))
            qset = set(q_terms.keys())
            matched_title = bool(qset & doc_title_tokens)
            matched_path = bool(qset & doc_path_tokens)
            matched_content = any((t in doc.get('tokens', [])) for t in qset)
            folder_path = Path(doc['path']).parent.as_posix()
        if folder_path == '.':
            folder_path = ''
        folder_dir = NOTES_DIR / folder_path if folder_path else NOTES_DIR
        folder_files = [p for p in folder_dir.glob('*.md') if p.is_file()] if folder_dir.exists() else []
        results.append({
                "path": doc['path'],
                "title": doc['title'],
                "score": float(score),
                "snippet": snippet,
                "matched_title": matched_title,
                "matched_path": matched_path,
                "matched_content": matched_content,
                "folder": folder_path,
                "folder_has_multiple_files": len(folder_files) > 1,
            })
        return results

    def _make_snippet(self, content: str, tokens: List[str], window: int = 80):
        low = content.lower()
        # find first occurrence of any token
        pos = -1
        found = None
        for t in tokens:
            p = low.find(t)
            if p >= 0 and (pos == -1 or p < pos):
                pos = p
                found = t
        if pos == -1:
            # return start
            snippet = content[:window * 2]
            return snippet.replace('\n', ' ')
        start = max(0, pos - window)
        end = min(len(content), pos + window)
        s = content[start:end]
        # highlight all tokens
        def repl(m):
            return f"<mark>{m.group(0)}</mark>"
        pattern = re.compile(r"(" + r"|".join(re.escape(t) for t in set(tokens)) + r")", re.I)
        s = pattern.sub(repl, s)
        return s.replace('\n', ' ')


# build global index
SEARCH_INDEX = None
try:
    SEARCH_INDEX = SearchIndex(NOTES_DIR)
except Exception:
    SEARCH_INDEX = None


def render_markdown(text: str) -> str:
    # Keep dollar delimiters in the markdown output and let the client
    # render math using KaTeX auto-render. Avoid server-side replacement
    # which can be fragile and may be escaped by the markdown renderer.
    return markdown2.markdown(
        text,
        extras=["fenced-code-blocks", "tables", "strike", "task_list", "cuddled-lists"],
    )


@app.route("/")
def index():
    notes = iter_markdown_files(NOTES_DIR)
    tree = build_tree(NOTES_DIR)
    return render_template("index.html", notes=notes, tree=tree, current_note=None)


@app.route("/note/<path:note_path>")
def view_note(note_path: str):
    candidate = (NOTES_DIR / note_path).resolve()
    if not str(candidate).startswith(str(NOTES_DIR.resolve())):
        abort(403)
    if not candidate.exists() or not candidate.is_file():
        abort(404)
    text = read_note(candidate)
    html = render_markdown(text)
    notes = iter_markdown_files(NOTES_DIR)
    tree = build_tree(NOTES_DIR)
    return render_template("index.html", notes=notes, tree=tree, current_note=candidate.relative_to(NOTES_DIR).as_posix(), content=html, title=candidate.stem)


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    results = []
    if query:
        if SEARCH_INDEX:
            results = SEARCH_INDEX.search(query, top_n=50)
        else:
            # fallback simple search
            q = query.lower()
            for path in iter_markdown_files(NOTES_DIR):
                text = read_note(path).lower()
                if q in text:
                    results.append({"path": path.relative_to(NOTES_DIR).as_posix(), "title": path.stem, "snippet": text[:200]})
    return render_template("search_results.html", query=query, results=results)


@app.route('/reindex')
def reindex():
    global SEARCH_INDEX
    try:
        SEARCH_INDEX = SearchIndex(NOTES_DIR)
        return "reindexed", 200
    except Exception as e:
        return str(e), 500


@app.route('/api/search')
def api_search():
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify([])
    results = []
    if SEARCH_INDEX:
        results = SEARCH_INDEX.search(q, top_n=20)
    else:
        qlow = q.lower()
        for path in iter_markdown_files(NOTES_DIR):
            text = read_note(path)
            if qlow in text.lower():
                results.append({"path": path.relative_to(NOTES_DIR).as_posix(), "title": path.stem, "snippet": text[:200]})
    return jsonify(results)


@app.route('/folder/<path:folder_path>')
def view_folder(folder_path: str):
    # show files under a folder (relative to NOTES_DIR)
    base = (NOTES_DIR / folder_path).resolve()
    if not str(base).startswith(str(NOTES_DIR.resolve())):
        abort(403)
    if not base.exists() or not base.is_dir():
        abort(404)
    files = []
    for p in sorted(base.glob('*.md')):
        files.append({
            'name': p.stem,
            'path': p.relative_to(NOTES_DIR).as_posix(),
        })
    # also collect subfolders
    subfolders = [d.name for d in sorted(base.iterdir()) if d.is_dir()]
    return render_template('folder.html', folder=folder_path, files=files, subfolders=subfolders)


@app.route("/static/<path:filename>")
def static_files(filename: str):
    return send_from_directory(STATIC_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
