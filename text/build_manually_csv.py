import pandas as pd

f = open('manually.txt', 'r')
lines = f.readlines()
data = []
for l in lines:
    parts = l.strip().split('/')
    subject = parts[4]
    emotion = parts[5]
    sentence = parts[6]
    path = l.strip().replace('../', '').replace('.txt', '.avi')
    data.append([subject, emotion, sentence, path])

df = pd.DataFrame(data, columns=['subject','emotion','sentence','path'])
df.to_csv('data/csvs/manually.csv', index=False)

