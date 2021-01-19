import numpy as np
import pandas as pd

features = np.load('data/data_selected_5.npy', allow_pickle=True)
df = pd.DataFrame(features)
print(df.describe())