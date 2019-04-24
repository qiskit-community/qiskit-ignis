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
    subprocess.call(['pip', 'install', '-U',
                     'git+https://github.com/Qiskit/qiskit-aer.git',
                     '--install-option', '--', '--install-option',
                     '-DCMAKE_CXX_COMPILER=g++-7'])
