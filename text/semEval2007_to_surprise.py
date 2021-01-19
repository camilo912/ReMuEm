import numpy as np
import pandas as pd

df = pd.read_csv('data/AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.emotions.gold', sep=' ', index_col=0, header=None)
df.index = np.arange(len(df))
fname = 'data/AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.xml'
f = open(fname, 'r')
lines = f.readlines()
lines = lines[1:-1]
tmp = [l[l.index('>')+1:l.rindex('<')] for l in lines]
df_text = pd.DataFrame(tmp, columns=['text'])
df_text.index = np.arange(len(df_text))
df_text = df_text[df.idxmax(axis=1)==6]
f.close()

df_2 = pd.read_csv('data/AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.emotions.gold', sep=' ', index_col=0, header=None)
df_2.index = np.arange(len(df_2))
fname = 'data/AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.xml'
f = open(fname, 'r')
lines = f.readlines()
lines = lines[1:-1]
tmp = [l[l.index('>')+1:l.rindex('<')] for l in lines]
df_text_2 = pd.DataFrame(tmp, columns=['text'])
df_text_2.index = np.arange(len(df_text_2))
df_text_2 = df_text_2[df_2.idxmax(axis=1)==6]

df_text = pd.concat((df_text,df_text_2), axis=0, ignore_index=True)

df_text.to_csv('data/csvs/semeval2007_surprise.csv', index=False)
f.close()