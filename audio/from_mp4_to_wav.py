import glob
import subprocess
import os

filepaths = glob.glob('../data/mosei/Raw/Videos/Segmented/Combined/*.mp4')
for path in filepaths:
    dest = path.replace('.mp4', '.wav').replace('Videos', 'Audio')
    if(not os.path.isfile(dest)):
        print('converting: '+path)
        subprocess.check_call(['ffmpeg', '-i', path, dest])