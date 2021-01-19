import pandas as pd
import numpy as np

df = pd.read_csv('data/senticnet5.csv', header=0, index_col=0)
df = df[['primary_mood', 'secondary_mood', 'pleasantness_value', 'attention_value', 'sensitivity_value', 'aptitude_value', 'polarity_value']]
df2 = pd.read_csv('data/affectivespace-csv/affectivespace.csv', header=None)
df2[0] = df2[0].apply(lambda x: str(x).lower())
df2 = df2.set_index(0)
as_list = df2.index.tolist()
idx = as_list.index('conspirac_theorist')
as_list[idx] = 'conspiracy_theorist'
df2.index = as_list

id1 = set(df.index)
id2 = set(df2.index)
valids = list(id1.intersection(id2))

df = df.loc[df.index.isin(valids), :]
df2 = df2.loc[df2.index.isin(valids), :]

df = df.merge(df2, left_index=True, right_index=True)

df.to_csv('data/mixed.csv')
