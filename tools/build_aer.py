#!/usr/bin/python3

import os
import shutil
import subprocess
import sys
import tempfile



if os.name == 'nt' or sys.platform == 'darwin':
    subprocess.call(['pip', 'install', '-U',
                     'git+https://github.com/Qiskit/qiskit-aer.git'])

if sys.platform == 'linux' or sys.platform == 'linux2':
    git_dir = tempfile.mkdtemp()
    try:
        subprocess.call(['git', 'clone',
                         'https://github.com/Qiskit/qiskit-aer.git', git_dir])
        subprocess.call([sys.executable, 'setup.py', 'install', '--',
                         '-DCMAKE_CXX_COMPILER=g++-7'], cwd=git_dir)
    finally:
        shutil.rmtree(git_dir)
