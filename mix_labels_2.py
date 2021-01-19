import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os

dataset_audio = 'iemocap'
dataset_face = 'hci'
dataset_text = 'fb'

if(not os.path.isdir('data/mix2')):
    os.mkdir('data/mix2')

data_audio = np.load('audio/data/'+dataset_audio+'/test_data.npy')
preds_audio = np.load('audio/data/'+dataset_audio+'/test_preds.npy')
labels_audio = np.load('audio/data/'+dataset_audio+'/test_labels.npy')
data_face = np.load('face/data/'+dataset_face+'/test_data.npy')
preds_face = np.load('face/data/'+dataset_face+'/test_preds.npy')
labels_face = np.load('face/data/'+dataset_face+'/test_labels.npy')
data_text = np.load('text/data/'+dataset_text+'/test_data.npy')
preds_text = np.load('text/data/'+dataset_text+'/test_preds.npy')
labels_text = np.load('text/data/'+dataset_text+'/test_labels.npy')

print(data_audio.shape)
print(preds_audio.shape)
print(labels_audio.shape)
print(data_face.shape)
print(preds_face.shape)
print(labels_face.shape)
print(data_text.shape)
print(preds_text.shape)
print(labels_text.shape)

ps = [0]
ps.append(data_audio.shape[1])
ps.append(ps[-1]+data_face.shape[1])
ps.append(ps[-1]+data_text.shape[1])
np.save('data/mix2/ps.npy', ps)

epsilon = 0.1

availables_face = list(range(len(data_face)))
availables_text = list(range(len(data_text)))
# taken_face = []
# taken_text = []
data = []
preds = []
labels = []
cont = 0
for i in range(len(data_audio)):
    data_i = list(data_audio[i])
    preds_i = [preds_audio[i]]
    labels_i = [labels_audio[i]]

    taken_face = None
    for j in availables_face:
        if(np.linalg.norm((labels_i[0]-labels_face[j]),2) < epsilon):
            data_i.extend(data_face[j])
            preds_i.append(preds_face[j])
            labels_i.append(labels_face[j])
            taken_face = j
            break
    if(taken_face is not None):
        availables_face.remove(taken_face)
    else:
        data_i.extend([np.nan for _ in range(data_face.shape[1])])
        preds_i.append([np.nan, np.nan])
        labels_i.append([np.nan, np.nan])
    
    taken_text = None
    for k in availables_text:
        if(np.linalg.norm((labels_i[0]-labels_text[k]),2) < epsilon):# or np.linalg.norm((labels_i[1]-labels_text[k]),2)):
            data_i.extend(data_text[k])
            preds_i.append(preds_text[k])
            labels_i.append(labels_text[k])
            taken_text = k
            break
    if(taken_text is not None):
        availables_text.remove(taken_text)
    else:
        data_i.extend([np.nan for _ in range(data_text.shape[1])])
        preds_i.append([np.nan, np.nan])
        labels_i.append([np.nan, np.nan])
    if(taken_face is not None and taken_text is not None):
        cont += 1
    data.append(data_i)
    preds.append(preds_i)
    labels.append(np.nanmean(np.array(labels_i), axis=0))

print("total completos:", cont)
print(len(availables_face))
print(len(availables_text))

for i in availables_face:
    data_i = [np.nan for _ in range(data_audio.shape[1])] + list(data_face[i])
    preds_i = [[np.nan, np.nan], preds_face[i]]
    labels_i = [[np.nan, np.nan], labels_face[i]]

    taken_text = None
    for j in availables_text:
        if(np.linalg.norm((labels_i[1]-labels_text[j]),2) < epsilon):
            data_i.extend(data_text[j])
            preds_i.append(preds_text[j])
            labels_i.append(labels_text[j])
            taken_text = j
            break
    if(taken_text is not None):
        availables_text.remove(taken_text)
    else:
        data_i.extend([np.nan for _ in range(data_text.shape[1])])
        preds_i.append([np.nan, np.nan])
        labels_i.append([np.nan, np.nan])
    data.append(data_i)
    preds.append(preds_i)
    labels.append(np.nanmean(np.array(labels_i), axis=0))

print(len(availables_text))

for i in availables_text:
    data_i = [np.nan for _ in range(data_audio.shape[1])] + [np.nan for _ in range(data_face.shape[1])] + list(data_text[i])
    preds_i = [[np.nan, np.nan], [np.nan, np.nan], preds_text[i]]
    labels_i = labels_text[i]
    data.append(data_i)
    preds.append(preds_i)
    labels.append(labels_i)

data = np.array(data)
preds = np.array(preds)
labels = np.array(labels)

print(data.shape)
print(preds.shape)
print(labels.shape)

np.save('data/mix2/data.npy', data)
np.save('data/mix2/preds.npy', preds)
np.save('data/mix2/labels.npy', labels)

# split
data_train, data_test, preds_train, preds_test, idxs_train, idxs_test = train_test_split(data, preds, range(len(data)), test_size=0.2)
labels_train = labels[idxs_train]
labels_test = labels[idxs_test]
data_train, data_val, preds_train, preds_val, idxs_train, idxs_val = train_test_split(data_train, preds_train, range(len(data_train)), test_size=0.5)
labels_val = labels_train[idxs_val]
labels_train = labels_train[idxs_train]


# train
np.save('data/mix2/data_train.npy', data_train)
np.save('data/mix2/preds_train.npy', preds_train)
np.save('data/mix2/labels_train.npy', labels_train)

# val
np.save('data/mix2/data_val.npy', data_val)
np.save('data/mix2/preds_val.npy', preds_val)
np.save('data/mix2/labels_val.npy', labels_val)

# test
np.save('data/mix2/data_test.npy', data_test)
np.save('data/mix2/preds_test.npy', preds_test)
np.save('data/mix2/labels_test.npy', labels_test)