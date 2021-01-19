import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

labels = np.load('audio/data/iemocap/test_labels.npy').T
fig, ax = plt.subplots()
plt.plot(-1, 0, 'wo', marker=".", markersize=0)
plt.plot(0, -1, 'wo', marker=".", markersize=0)
plt.plot(1, 0, 'wo', marker=".", markersize=0)
plt.plot(0, 1, 'wo', marker=".", markersize=0)
plt.plot(0, 0, 'ko', marker="+")
plt.plot(labels[:,1], labels[:,0], 'bo', label='labels', markersize=2)
ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False))
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
# plt.show()

labels = np.load('face/data/hci/test_labels.npy').T
print(pd.Series(labels[:,0]).describe())
print(pd.Series(labels[:,1]).describe())
fig, ax = plt.subplots()
plt.plot(-1, 0, 'wo', marker=".", markersize=0)
plt.plot(0, -1, 'wo', marker=".", markersize=0)
plt.plot(1, 0, 'wo', marker=".", markersize=0)
plt.plot(0, 1, 'wo', marker=".", markersize=0)
plt.plot(0, 0, 'ko', marker="+")
plt.plot(labels[:,1], labels[:,0], 'bo', label='labels', markersize=2)
ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False))
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
# plt.show()

labels = np.load('text/data/fb/test_labels.npy').T
fig, ax = plt.subplots()
plt.plot(-1, 0, 'wo', marker=".", markersize=0)
plt.plot(0, -1, 'wo', marker=".", markersize=0)
plt.plot(1, 0, 'wo', marker=".", markersize=0)
plt.plot(0, 1, 'wo', marker=".", markersize=0)
plt.plot(0, 0, 'ko', marker="+")
plt.plot(labels[:,1], labels[:,0], 'bo', label='labels', markersize=2)
ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False))
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.show()