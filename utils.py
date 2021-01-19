import numpy as np
from keras.utils import Sequence
from keras import backend as K
import random

# values extracted from: Russell, J. A., & Mehrabian, A. (1977). Evidence for a three-factor theory of emotions. Journal of research in Personality, 11(3), 273-294.
def get_map():
    return {'anger':[-0.51, 0.59], 'disgust':[-0.6, 0.35], 'fear':[-0.64, 0.6], 'happiness':[0.81, 0.51], 'sadness':[-0.63, -0.27], 'surprise':[0.4, 0.67]}

def get_closest(arousal, valence):
    vec = np.array([arousal, valence], dtype = np.flaot32)
    emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
    mapa = get_map()

    distances = []
    for emotion in emotions:
        distances.append(np.linalg.norm(vec-np.array(mapa[emotion]), 2))

    return emotions[np.argmin(distances)]

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()))

def r2_keras_inv(y_true, y_pred):
    return 1-r2_keras(y_true, y_pred)

class CustomSequence(Sequence):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_batches(X_train, X_test, Y_train, Y_test, batch_size):
    #create batches: f = full, t = two modalities, o = one modality
    X_train_f, X_train_t, X_train_o, Y_train_f, Y_train_t, Y_train_o = [], [], [], [], [], []
    for x,y in zip(X_train, Y_train):
        if(x.shape[1]==3):
            X_train_f.append(x)
            Y_train_f.append(y)
        elif(x.shape[1]==2):
            X_train_t.append(x)
            Y_train_t.append(y)
        elif(x.shape[1]==1):
            X_train_o.append(x)
            Y_train_o.append(y)
        else:
            raise Exception('Invalid length: '+str(x.shape[1]))
    X_train_f = np.array(X_train_f).squeeze()
    X_train_t = np.array(X_train_t).squeeze()
    X_train_o = np.array(X_train_o).reshape(-1,1,2)
    Y_train_f = np.array(Y_train_f).squeeze()
    Y_train_t = np.array(Y_train_t).squeeze()
    Y_train_o = np.array(Y_train_o).squeeze()

    X_test_f, X_test_t, X_test_o, Y_test_f, Y_test_t, Y_test_o = [], [], [], [], [], []
    for x,y in zip(X_test, Y_test):
        if(x.shape[1]==3):
            X_test_f.append(x)
            Y_test_f.append(y)
        elif(x.shape[1]==2):
            X_test_t.append(x)
            Y_test_t.append(y)
        elif(x.shape[1]==1):
            X_test_o.append(x)
            Y_test_o.append(y)
        else:
            raise Exception('Invalid length: '+str(x.shape[1]))
    X_test_f = np.array(X_test_f).squeeze()
    X_test_t = np.array(X_test_t).squeeze()
    X_test_o = np.array(X_test_o).reshape(-1,1,2)
    Y_test_f = np.array(Y_test_f).squeeze()
    Y_test_t = np.array(Y_test_t).squeeze()
    Y_test_o = np.array(Y_test_o).squeeze()

    batches_X_train = []
    batches_Y_train = []
    for i in range(100):
        idxs = random.sample(range(len(X_train_f)), min(batch_size, len(X_train_f)))
        batches_X_train.append(X_train_f[idxs])
        batches_Y_train.append(Y_train_f[idxs])
    for i in range(100):
        idxs = random.sample(range(len(X_train_t)), min(batch_size, len(X_train_t)))
        batches_X_train.append(X_train_t[idxs])
        batches_Y_train.append(Y_train_t[idxs])
    for i in range(100):
        idxs = random.sample(range(len(X_train_o)), min(batch_size, len(X_train_o)))
        batches_X_train.append(X_train_o[idxs])
        batches_Y_train.append(Y_train_o[idxs])

    batches_X_test = []
    batches_Y_test = []
    for i in range(100):
        idxs = random.sample(range(len(X_test_f)), min(batch_size, len(X_test_f)))
        batches_X_test.append(X_test_f[idxs])
        batches_Y_test.append(Y_test_f[idxs])
    for i in range(100):
        idxs = random.sample(range(len(X_test_t)), min(batch_size, len(X_test_t)))
        batches_X_test.append(X_test_t[idxs])
        batches_Y_test.append(Y_test_t[idxs])
    for i in range(100):
        idxs = random.sample(range(len(X_test_o)), min(batch_size, len(X_test_o)))
        batches_X_test.append(X_test_o[idxs])
        batches_Y_test.append(Y_test_o[idxs])
    
    return batches_X_train, batches_X_test, batches_Y_train, batches_Y_test