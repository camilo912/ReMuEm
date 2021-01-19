import numpy as np

labels = np.load('data/mix/labels.npy')
for i in range(len(labels)):
    n = np.linalg.norm(labels[i], 2)
    if(n > 1):
        labels[i] = labels[i]/n

np.save('data/mix/labels_n.npy', labels)