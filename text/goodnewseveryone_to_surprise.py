import pandas as pd
import json

fname = 'data/GoodNewsEveryone/release_review.jsonl'
f = open(fname, 'r')
lines = [json.loads(jline) for jline in f.readlines()]
lines = [l['headline'] for l in lines if 'surprise' in l['annotations']['dominant_emotion']['gold']]
df = pd.DataFrame(lines, columns = ['text'])
df.to_csv('data/csvs/goodnewseveryone_surprise.csv', index=False)
f.close()