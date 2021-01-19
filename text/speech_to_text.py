import numpy as np
import speech_recognition as sr
import glob
import os

r = sr.Recognizer()
filepaths = glob.glob('../data/enterface/enterface database/*/*/*/*.avi')
for path in filepaths:
    src = path.replace('.avi', '.wav')
    dst = path.replace('.avi', '.txt')
    if(not os.path.isfile(dst)):# and not dst == '../data/enterface/enterface database/subject 3/surprise/sentence 1/s_3_su_1.txt'):
        entrada = sr.AudioFile(src)
        with entrada as source:
            audio = r.record(source)
            try:
                text = r.recognize_google(audio)
                with open(dst, 'w') as f:
                    f.write(text)
            except :
                print(dst)
                pass