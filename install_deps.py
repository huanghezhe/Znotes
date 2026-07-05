import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'flask', 'markdown2'])
print('installed')
