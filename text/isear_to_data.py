import numpy as np
import pandas as pd

df = pd.read_csv('data/py_isear_dataset/isear.csv', sep='|', header=0, usecols=['Field1', 'SIT'])
df.columns = ['emotion', 'text']
df_surp = pd.read_csv('data/csvs/semeval2007_surprise.csv', header=0)
# df_surp = pd.read_csv('data/csvs/goodnewseveryone_surprise.csv', header=0)

df_surp['emotion'] = 'surprise'
df = pd.concat((df, df_surp), axis=0, ignore_index=True)

emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

df = df[df['emotion'].isin(emotions)]
df.loc[df['emotion']=='joy', 'emotion'] = 'happiness'
df.to_csv('data/csvs/isear_data.csv', index=False)