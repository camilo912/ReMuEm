import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from sklearn.model_selection import train_test_split

dataset_audio = 'iemocap'
dataset_face = 'hci'
dataset_text = 'fb'

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
np.save('data/mix/ps.npy', ps)

epsilon = 0.05

# all
data = []
labels = []
preds = []

def parallel_func(i):
    data = []
    labels = []
    preds = []
    for j in range(len(labels_face)):
        for k in range(len(labels_text)):
            a_f = np.linalg.norm(labels_audio[i]-labels_face[j], 2)
            f_t = np.linalg.norm(labels_text[k]-labels_face[j], 2)
            a_t = np.linalg.norm(labels_text[k]-labels_audio[i], 2)
            # if((a_f<epsilon and f_t<epsilon) or (a_f<epsilon and a_t<epsilon) or (a_t<epsilon and f_t<epsilon)):
            if(a_f<epsilon and f_t<epsilon and a_t<epsilon):
                multi = np.append(np.append(labels_audio[[i]], labels_face[[j]], axis=0), labels_text[[k]], axis=0)
                med = np.mean(multi,axis=0)
                data_i = np.concatenate((data_audio[i], data_face[j], data_text[k]), axis=0)
                data.append(data_i)
                labels.append(med)
                multi = np.append(np.append(preds_audio[[i]], preds_face[[j]], axis=0), preds_text[[k]], axis=0)
                preds.append(multi)
    return [data, labels, preds]
    


# for i in range(len(labels_audio)):
pool = mp.Pool(mp.cpu_count()-3)
results = list(tqdm(pool.imap(parallel_func, range(len(labels_audio)))))
pool.close()

for r in results:
    if(len(r[0]) > 0):
        data.extend(r[0])
        labels.extend(r[1])
        preds.extend(r[2])
    

data = np.array(data)
preds = np.array(preds)
labels = np.array(labels)

print(data.shape)
print(preds.shape)
print(labels.shape)

np.save('data/mix/data.npy', data)
np.save('data/mix/preds.npy', preds)
np.save('data/mix/labels.npy', labels)

# split before mixing
data_audio_train, data_audio_test, preds_audio_train, preds_audio_test, idxs_train, idxs_test = train_test_split(data_audio, preds_audio, range(len(preds_audio)), test_size=0.2)
data_audio_train, data_audio_val, preds_audio_train, preds_audio_val, idxs_train, idxs_val = train_test_split(data_audio_train, preds_audio_train, idxs_train, test_size=0.5)
labels_audio_train = labels_audio[idxs_train]
labels_audio_val = labels_audio[idxs_val]
labels_audio_test = labels_audio[idxs_test]
data_face_train, data_face_test, preds_face_train, preds_face_test, idxs_train, idxs_test = train_test_split(data_face, preds_face, range(len(preds_face)), test_size=0.2)
data_face_train, data_face_val, preds_face_train, preds_face_val, idxs_train, idxs_val = train_test_split(data_face_train, preds_face_train, idxs_train, test_size=0.5)
labels_face_train = labels_face[idxs_train]
labels_face_val = labels_face[idxs_val]
labels_face_test = labels_face[idxs_test]
data_text_train, data_text_test, preds_text_train, preds_text_test, idxs_train, idxs_test = train_test_split(data_text, preds_text, range(len(preds_text)), test_size=0.2)
data_text_train, data_text_val, preds_text_train, preds_text_val, idxs_train, idxs_val = train_test_split(data_text_train, preds_text_train, idxs_train, test_size=0.5)
labels_text_train = labels_text[idxs_train]
labels_text_val = labels_text[idxs_val]
labels_text_test = labels_text[idxs_test]

# train
def parallel_func_train(i):
    data = []
    labels = []
    preds = []
    for j in range(len(labels_face_train)):
        for k in range(len(labels_text_train)):
            a_f = np.linalg.norm(labels_audio_train[i]-labels_face_train[j], 2)
            f_t = np.linalg.norm(labels_text_train[k]-labels_face_train[j], 2)
            a_t = np.linalg.norm(labels_text_train[k]-labels_audio_train[i], 2)
            # if((a_f<epsilon and f_t<epsilon) or (a_f<epsilon and a_t<epsilon) or (a_t<epsilon and f_t<epsilon)):
            if(a_f<epsilon and f_t<epsilon and a_t<epsilon):
                multi = np.append(np.append(labels_audio_train[[i]], labels_face_train[[j]], axis=0), labels_text_train[[k]], axis=0)
                med = np.mean(multi,axis=0)
                data_i = np.concatenate((data_audio_train[i], data_face_train[j], data_text_train[k]), axis=0)
                data.append(data_i)
                labels.append(med)
                multi = np.append(np.append(preds_audio_train[[i]], preds_face_train[[j]], axis=0), preds_text_train[[k]], axis=0)
                preds.append(multi)
    return [data, labels, preds]
    
pool = mp.Pool(mp.cpu_count()-3)
results = list(tqdm(pool.imap(parallel_func_train, range(len(labels_audio_train)))))
pool.close()

data_train = []
labels_train = []
preds_train = []
for r in results:
    if(len(r[0]) > 0):
        data_train.extend(r[0])
        labels_train.extend(r[1])
        preds_train.extend(r[2])
    

data_train = np.array(data_train)
preds_train = np.array(preds_train)
labels_train = np.array(labels_train)

print(data_train.shape)
print(preds_train.shape)
print(labels_train.shape)

np.save('data/mix/data_train.npy', data_train)
np.save('data/mix/preds_train.npy', preds_train)
np.save('data/mix/labels_train.npy', labels_train)

# val
def parallel_func_val(i):
    data = []
    labels = []
    preds = []
    for j in range(len(labels_face_val)):
        for k in range(len(labels_text_val)):
            a_f = np.linalg.norm(labels_audio_val[i]-labels_face_val[j], 2)
            f_t = np.linalg.norm(labels_text_val[k]-labels_face_val[j], 2)
            a_t = np.linalg.norm(labels_text_val[k]-labels_audio_val[i], 2)
            # if((a_f<epsilon and f_t<epsilon) or (a_f<epsilon and a_t<epsilon) or (a_t<epsilon and f_t<epsilon)):
            if(a_f<epsilon and f_t<epsilon and a_t<epsilon):
                multi = np.append(np.append(labels_audio_val[[i]], labels_face_val[[j]], axis=0), labels_text_val[[k]], axis=0)
                med = np.mean(multi,axis=0)
                data_i = np.concatenate((data_audio_val[i], data_face_val[j], data_text_val[k]), axis=0)
                data.append(data_i)
                labels.append(med)
                multi = np.append(np.append(preds_audio_val[[i]], preds_face_val[[j]], axis=0), preds_text_val[[k]], axis=0)
                preds.append(multi)
    return [data, labels, preds]
    
pool = mp.Pool(mp.cpu_count()-3)
results = list(tqdm(pool.imap(parallel_func_val, range(len(labels_audio_val)))))
pool.close()

data_val = []
labels_val = []
preds_val = []
for r in results:
    if(len(r[0]) > 0):
        data_val.extend(r[0])
        labels_val.extend(r[1])
        preds_val.extend(r[2])
    

data_val = np.array(data_val)
preds_val = np.array(preds_val)
labels_val = np.array(labels_val)

print(data_val.shape)
print(preds_val.shape)
print(labels_val.shape)

np.save('data/mix/data_val.npy', data_val)
np.save('data/mix/preds_val.npy', preds_val)
np.save('data/mix/labels_val.npy', labels_val)

# test
def parallel_func_test(i):
    data = []
    labels = []
    preds = []
    for j in range(len(labels_face_test)):
        for k in range(len(labels_text_test)):
            a_f = np.linalg.norm(labels_audio_test[i]-labels_face_test[j], 2)
            f_t = np.linalg.norm(labels_text_test[k]-labels_face_test[j], 2)
            a_t = np.linalg.norm(labels_text_test[k]-labels_audio_test[i], 2)
            # if((a_f<epsilon and f_t<epsilon) or (a_f<epsilon and a_t<epsilon) or (a_t<epsilon and f_t<epsilon)):
            if(a_f<epsilon and f_t<epsilon and a_t<epsilon):
                multi = np.append(np.append(labels_audio_test[[i]], labels_face_test[[j]], axis=0), labels_text_test[[k]], axis=0)
                med = np.mean(multi,axis=0)
                data_i = np.concatenate((data_audio_test[i], data_face_test[j], data_text_test[k]), axis=0)
                data.append(data_i)
                labels.append(med)
                multi = np.append(np.append(preds_audio_test[[i]], preds_face_test[[j]], axis=0), preds_text_test[[k]], axis=0)
                preds.append(multi)
    return [data, labels, preds]
    
pool = mp.Pool(mp.cpu_count()-3)
results = list(tqdm(pool.imap(parallel_func_test, range(len(labels_audio_test)))))
pool.close()

data_test = []
labels_test = []
preds_test = []
for r in results:
    if(len(r[0]) > 0):
        data_test.extend(r[0])
        labels_test.extend(r[1])
        preds_test.extend(r[2])
    

data_test = np.array(data_test)
preds_test = np.array(preds_test)
labels_test = np.array(labels_test)

print(data_test.shape)
print(preds_test.shape)
print(labels_test.shape)

np.save('data/mix/data_test.npy', data_test)
np.save('data/mix/preds_test.npy', preds_test)
np.save('data/mix/labels_test.npy', labels_test)