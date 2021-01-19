import glob
import subprocess
import os

filepaths = glob.glob('../data/enterface/enterface database/*/*/*/*.avi')
for path in filepaths:
    dest = path.replace('.avi', '.wav')
    if(not os.path.isfile(dest)):
        print('converting: '+path)
        subprocess.check_call(['sudo', 'ffmpeg', '-i', path, dest])