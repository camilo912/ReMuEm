import numpy as np
import pandas as pd
import os
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from functools import partial
from matplotlib import pyplot as plt
from keras.metrics import RootMeanSquaredError
from keras import backend as K
from keras.regularizers import l1, l2
from utils import r2_keras
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
import time

def custom_loss(y_true, y_pred):
    # y_true = K.cast(y_true, dtype='float32')
    # y_pred = K.cast(y_pred, dtype='float32')
    # loss = K.mean(K.square(y_true - y_pred))
    #loss += K.square(K.std(y_true)-K.std(y_pred))
    # loss = K.abs(K.std(y_true)-K.std(y_pred))/K.std(y_true)
    # return loss
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return SS_res/(SS_tot + K.epsilon())
    # return (SS_res/(SS_tot + K.epsilon()))/K.size(y_pred)

def create_model(drop_rate, lr, n_neurons, reg_rate):
    model = Sequential()
    model.add(Dense(n_neurons, activation='relu', kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate), activity_regularizer=l2(reg_rate)))
    # model.add(Dense(n_neurons, activation='relu', kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate), activity_regularizer=l2(reg_rate)))
    # model.add(Dense(n_neurons, activation='relu', kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate), activity_regularizer=l2(reg_rate)))
    model.add(Dropout(drop_rate))
    # model.add(Dense(int(n_neurons/2), activation='relu', kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate), activity_regularizer=l2(reg_rate)))
    # model.add(Dense(int(n_neurons/2), activation='relu', kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate), activity_regularizer=l2(reg_rate)))
    # model.add(Dense(int(n_neurons/2), activation='relu', kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate), activity_regularizer=l2(reg_rate)))
    # model.add(Dropout(drop_rate))
    # model.add(Dense(int(n_neurons/4), activation='relu', kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate), activity_regularizer=l2(reg_rate)))
    # model.add(Dropout(drop_rate))
    # model.add(Dense(int(n_neurons/2), activation='relu', kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate), activity_regularizer=l2(reg_rate)))
    # model.add(Dense(int(n_neurons/2), activation='relu', kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate), activity_regularizer=l2(reg_rate)))
    # model.add(Dropout(drop_rate))
    # model.add(Dense(n_neurons, activation='relu', kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate), activity_regularizer=l2(reg_rate)))
    # model.add(Dropout(drop_rate))
    # model.add(Dense(n_neurons, activation='relu', kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate), activity_regularizer=l2(reg_rate)))
    # model.add(Dropout(drop_rate))
    # model.add(Dense(int(n_neurons/2), activation='relu', kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate), activity_regularizer=l2(reg_rate)))
    # model.add(Dropout(drop_rate))
    model.add(Dense(2, activation='tanh'))
    opt = Adam(learning_rate=lr)
    model.compile(loss=custom_loss, optimizer=opt, metrics=[RootMeanSquaredError(), r2_keras])
    return model

def train(X_train, X_test, Y_train, Y_test, verbose, batch_size, drop_rate, lr, n_epochs, n_neurons, reg_rate):
    model = create_model(drop_rate, lr, n_neurons, reg_rate)
    history = model.fit(X_train, Y_train, validation_data=[X_test, Y_test], epochs=n_epochs, verbose=verbose, batch_size=batch_size)

    losses_tr = [x for x in history.history['loss']]
    losses_te = [x for x in history.history['val_loss']]
    rmses_tr = [x for x in history.history['root_mean_squared_error']]
    rmses_te = [x for x in history.history['val_root_mean_squared_error']]
    r2s_tr = [x for x in history.history['r2_keras']]
    r2s_te = [x for x in history.history['val_r2_keras']]
    
    if(verbose):
        plt.subplot(3,1,1)
        plt.plot(losses_tr, label='loss training')
        plt.plot(losses_te, label='loss testing')
        plt.legend()
        plt.subplot(3,1,2)
        plt.plot(rmses_tr, label='rmse training')
        plt.plot(rmses_te, label='rmse testing')
        plt.legend()
        plt.subplot(3,1,3)
        plt.plot(r2s_tr, label='r2 training')
        plt.plot(r2s_te, label='r2 testing')
        plt.legend()
        
    return model, losses_te[-1]

def _train(X_train, X_test, Y_train, Y_test, verbose, batch_size, drop_rate, lr, n_epochs, n_neurons, reg_rate):
    n_epochs = int(n_epochs)
    n_neurons = int(n_neurons)
    model, loss = train(X_train, X_test, Y_train, Y_test, verbose, batch_size, dorp_rate, lr, n_epochs, n_neurons, reg_rate)
    return 1-loss

def main():
    # load data

    # # mosei
    # # Y_audio = np.array([(0.1,0.2), (0.2,-0.3), (0.3,0.4), (-0.3,0.2), (0.5,0.1), (-0.21, -0.01), (-0.51, 0.15), (-0.15, -0.61)], dtype = np.float32)
    # # Y_face = np.array([(0.7,-0.5), (0.53,0.23), (-0.48,0.12), (0.31,-0.5), (0.37,-0.6), (1.5, 0.31), (0.26, -0.03), (0.13, 0.36)], dtype = np.float32)
    # # Y_text = np.array([(-0.5,-0.2), (0.72,-0.38), (-0.34,-0.48), (0.15,0.28), (0.69,-0.42), (0.2, -0.35), (-0.21, -0.01), (-0.12, -0.41)], dtype = np.float32)
    # # Y_true = np.array([(0.5,0.21), (0.4,0.19), (0.22,0.43), (-0.35,-0.2), (0.58,-0.18), (0.08, -0.25), (0.11, 0.21), (-0.33, 0.4)], dtype = np.float32)

    # Y_audio = np.load('audio/data/iemocap/preds.npy', allow_pickle=True)
    # Y_face = np.load('face/data/hci/preds.npy', allow_pickle=True)
    # Y_text = np.load('text/data/fb/preds.npy', allow_pickle=True)
    # Y_true = np.load('audio/data/iemocap/preds_labels.npy', allow_pickle=True)

    # ps = [0]
    # ps.append(Y_audio.shape[1])
    # ps.append(ps[-1]+Y_face.shape[1])
    # ps.append(ps[-1]+Y_text.shape[1])

    # Y_multi = np.concatenate((Y_audio, Y_face, Y_text), axis=1)

    # # split train and test (for this model from 25% to 50% for training, from 50% to 75% for testing)
    # size = int(len(Y_multi)*0.25)
    # Y_multi_train = Y_multi[size:size*2]
    # Y_multi_test = Y_multi[size*2:size*3]
    # Y_true_train = Y_true[size:size*2]
    # Y_true_test = Y_true[size*2:size*3]

    # mixed
    # ps = np.load('data/mix/ps.npy')
    # # X = np.load('data/mix/data.npy')
    # # Y = np.load('data/mix/labels.npy')

    # # data_train, data_test, labels_train, labels_test = train_test_split(X, Y, test_size=0.3)
    # data_train = np.load('data/mix/data_train.npy')
    # data_test = np.load('data/mix/data_test.npy')
    # labels_train = np.load('data/mix/labels_train.npy')
    # labels_test = np.load('data/mix/labels_test.npy')

    
    # # remove nans
    # # train
    # assert len(labels_train) == len(data_train)
    # idxs = [i for i in range(len(labels_train)) if np.isnan(data_train[i]).any()]
    # data_train = np.delete(data_train, idxs, axis=0)
    # labels_train = np.delete(labels_train, idxs, axis=0)
    # #test
    # assert len(labels_test) == len(data_test)
    # idxs = [i for i in range(len(labels_test)) if np.isnan(data_test[i]).any()]
    # data_test = np.delete(data_test, idxs, axis=0)
    # labels_test = np.delete(labels_test, idxs, axis=0)

    # X_train = []
    # Y_train = []
    # for X_i, Y_i in zip(data_train, labels_train):
    #     X_train.append(X_i.reshape(-1))
    #     Y_train.append(Y_i)
    #     for j in range(len(ps)-1):
    #         X_j = X_i.copy()
    #         X_jt = np.zeros((ps[-1],), dtype=np.float32)
    #         X_jt[ps[j]:ps[j+1]] = X_j[ps[j]:ps[j+1]]
    #         X_j[ps[j]:ps[j+1]] = [0.0 for _ in range(ps[j+1]-ps[j])]
    #         X_train.append(X_j)
    #         Y_train.append(Y_i)
    #         X_train.append(X_jt)
    #         Y_train.append(Y_i)
    
    # X_test = []
    # Y_test = []
    # for X_i, Y_i in zip(data_test, labels_test):
    #     X_test.append(X_i.reshape(-1))
    #     Y_test.append(Y_i)
    #     for j in range(len(ps)-1):
    #         X_j = X_i.copy()
    #         X_jt = np.zeros((ps[-1],), dtype=np.float32)
    #         X_jt[ps[j]:ps[j+1]] = X_j[ps[j]:ps[j+1]]
    #         X_j[ps[j]:ps[j+1]] = [0.0 for _ in range(ps[j+1]-ps[j])]
    #         X_test.append(X_j)
    #         Y_test.append(Y_i)
    #         X_test.append(X_jt)
    #         Y_test.append(Y_i)

    # X_train = np.array(X_train)
    # X_test = np.array(X_test)
    # Y_train = np.array(Y_train)
    # Y_test = np.array(Y_test)

    data_train = np.load('data/mix2/data_train.npy')
    data_val = np.load('data/mix2/data_val.npy')
    data_test = np.load('data/mix2/data_test.npy')
    Y_train = np.load('data/mix2/labels_train.npy')
    Y_val = np.load('data/mix2/labels_val.npy')
    Y_test = np.load('data/mix2/labels_test.npy')

    X_train = np.nan_to_num(data_train)
    X_val = np.nan_to_num(data_val)
    X_test = np.nan_to_num(data_test)

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_val.shape)
    print(Y_test.shape)

    # ***** train *****
    if(not os.path.isdir('data/models')):
        os.mkdir('data/models')
    if(not os.path.isdir('data/models/flf_nn')):
        os.mkdir('data/models/flf_nn')
    
    # # arousal
    # clf = RandomForestRegressor()
    # # clf = SVR(C=0.5)
    # clf.fit(X_train.reshape(-1, 376), Y_train[:,0])
    # preds_aro_train = clf.predict(X_train.reshape(-1, 376))
    # preds_aro_val = clf.predict(X_val.reshape(-1, 376))
    # t0 = time.time()
    # preds_aro_test = clf.predict(X_test.reshape(-1, 376))
    # print("time for predicting {} samples: {}".format(len(X_test), time.time()-t0))

    # preds_aro_train = np.clip(preds_aro_train, -1, 1)
    # preds_aro_val = np.clip(preds_aro_val, -1, 1)
    # preds_aro_test = np.clip(preds_aro_test, -1, 1)
    
    # print("r2 train aro:", r2_score(Y_train[:,0], preds_aro_train))
    # print("rmse train aro:", mean_squared_error(Y_train[:,0], preds_aro_train)**(1/2))
    # print("std train relative error:", (np.std(Y_train[:,0], axis=0)-np.std(preds_aro_train, axis=0))/np.std(Y_train[:,0], axis=0))
    # print("r2 val aro:", r2_score(Y_val[:,0], preds_aro_val))
    # print("rmse val aro:", mean_squared_error(Y_val[:,0], preds_aro_val)**(1/2))
    # print("std val relative error:", (np.std(Y_val[:,0], axis=0)-np.std(preds_aro_val, axis=0))/np.std(Y_val[:,0], axis=0))
    # print("r2 test aro:", r2_score(Y_test[:,0], preds_aro_test))
    # print("rmse test aro:", mean_squared_error(Y_test[:,0], preds_aro_test)**(1/2))
    # print("std test relative error:", (np.std(Y_test[:,0], axis=0)-np.std(preds_aro_test, axis=0))/np.std(Y_test[:,0], axis=0))

    # # valence
    # clf = RandomForestRegressor()
    # # clf = SVR(C=0.5)
    # clf.fit(X_train.reshape(-1, 376), Y_train[:,1])
    # preds_vale_train = clf.predict(X_train.reshape(-1, 376))
    # preds_vale_val = clf.predict(X_val.reshape(-1, 376))
    # t0 = time.time()
    # preds_vale_test = clf.predict(X_test.reshape(-1, 376))
    # print('\n\n')
    # print("time for predicting {} samples: {}".format(len(X_test), time.time()-t0))

    # preds_vale_train = np.clip(preds_vale_train, -1, 1)
    # preds_vale_val = np.clip(preds_vale_val, -1, 1)
    # preds_vale_test = np.clip(preds_vale_test, -1, 1)

    # print("r2 train vale:", r2_score(Y_train[:,1], preds_vale_train))
    # print("rmse train vale:", mean_squared_error(Y_train[:,1], preds_vale_train)**(1/2))
    # print("std train relative error:", (np.std(Y_train[:,1], axis=0)-np.std(preds_vale_train, axis=0))/np.std(Y_train[:,1], axis=0))
    # print("r2 val vale:", r2_score(Y_val[:,1], preds_vale_val))
    # print("rmse val vale:", mean_squared_error(Y_val[:,1], preds_vale_val)**(1/2))
    # print("std val relative error:", (np.std(Y_val[:,1], axis=0)-np.std(preds_vale_val, axis=0))/np.std(Y_val[:,1], axis=0))
    # print("r2 test vale:", r2_score(Y_test[:,1], preds_vale_test))
    # print("rmse test vale:", mean_squared_error(Y_test[:,1], preds_vale_test)**(1/2))
    # print("std test relative error:", (np.std(Y_test[:,1], axis=0)-np.std(preds_vale_test, axis=0))/np.std(Y_test[:,1], axis=0))

    # preds_train = np.concatenate((preds_aro_train.reshape(-1,1), preds_vale_train.reshape(-1,1)), axis=1)
    # preds_val = np.concatenate((preds_aro_val.reshape(-1,1), preds_vale_val.reshape(-1,1)), axis=1)
    # preds_test = np.concatenate((preds_aro_test.reshape(-1,1), preds_vale_test.reshape(-1,1)), axis=1)

    # clf = RandomForestRegressor()
    # # clf = PLSRegression(n_components=15)
    # clf.fit(X_train.reshape(-1, 376), Y_train)
    # preds_train = clf.predict(X_train.reshape(-1, 376))
    # preds_val = clf.predict(X_val.reshape(-1, 376))
    # preds_test = clf.predict(X_test.reshape(-1, 376))

    # preds_train = np.clip(preds_train, -1, 1)
    # preds_val = np.clip(preds_val, -1, 1)
    # preds_test = np.clip(preds_test, -1, 1)

    # hyper parameters
    batch_size = 32
    drop_rate = 0.5
    n_epochs = 125
    lr = 1e-4
    n_neurons = 256
    reg_rate = 0

    model, loss = train(X_train, X_val, Y_train, Y_val, 1, batch_size, drop_rate, lr, n_epochs, n_neurons, reg_rate)
    model.save('data/models/flf_nn/mlp_'+str(round(loss,4))+'_'+str(batch_size)+'_'+str(drop_rate)+'_'+str(lr)+'_'+str(n_epochs)+'_'+str(n_neurons)+'_'+str(reg_rate)+'.h5')

    preds_train = model.predict(X_train)
    preds_val = model.predict(X_val)
    t0 = time.time()
    preds_test = model.predict(X_test)
    print("time for predicting {} samples: {}".format(len(X_test), time.time()-t0))

    print('\n\n')
    print("r2 train:", r2_score(Y_train, preds_train))
    print('rmse train:', mean_squared_error(Y_train, preds_train)**(1/2))
    print("std train relative error:", (np.linalg.norm(np.std(Y_train, axis=0)-np.std(preds_train, axis=0),2)/np.linalg.norm(np.std(Y_train, axis=0), 2)))
    print("r2 val:", r2_score(Y_val, preds_val))
    print('rmse val:', mean_squared_error(Y_val, preds_val)**(1/2))
    print("std val relative error:", (np.linalg.norm(np.std(Y_val, axis=0)-np.std(preds_val, axis=0),2)/np.linalg.norm(np.std(Y_val, axis=0), 2)))
    print("r2 test:", r2_score(Y_test, preds_test))
    print('rmse test:', mean_squared_error(Y_test, preds_test)**(1/2))
    print("std test relative error:", (np.linalg.norm(np.std(Y_test, axis=0)-np.std(preds_test, axis=0),2)/np.linalg.norm(np.std(Y_test, axis=0), 2)))

    errors_val = Y_val - preds_val
    errors_test = Y_test - preds_test

    np.save('data/outputs/flf_zero_padding/val_preds.npy', preds_val)
    np.save('data/outputs/flf_zero_padding/val_data.npy', X_val)
    np.save('data/outputs/flf_zero_padding/val_errors.npy', errors_val)
    np.save('data/outputs/flf_zero_padding/val_labels.npy', Y_val)
    np.save('data/outputs/flf_zero_padding/test_preds.npy', preds_test)
    np.save('data/outputs/flf_zero_padding/test_data.npy', X_test)
    np.save('data/outputs/flf_zero_padding/test_errors.npy', errors_test)
    np.save('data/outputs/flf_zero_padding/test_labels.npy', Y_test)

    print("r2 train:", r2_score(Y_train, preds_train))
    print('rmse train:', mean_squared_error(Y_train, preds_train)**(1/2))
    print("std train relative error:", (np.linalg.norm(np.std(Y_train, axis=0)-np.std(preds_train, axis=0),2)/np.linalg.norm(np.std(Y_train, axis=0), 2)))
    print("r2 val:", r2_score(Y_val, preds_val))
    print('rmse val:', mean_squared_error(Y_val, preds_val)**(1/2))
    print("std val relative error:", (np.linalg.norm(np.std(Y_val, axis=0)-np.std(preds_val, axis=0),2)/np.linalg.norm(np.std(Y_val, axis=0), 2)))
    print("r2 test:", r2_score(Y_test, preds_test))
    print('rmse test:', mean_squared_error(Y_test, preds_test)**(1/2))
    print("std test relative error:", (np.linalg.norm(np.std(Y_test, axis=0)-np.std(preds_test, axis=0),2)/np.linalg.norm(np.std(Y_test, axis=0), 2)))

    fig, ax = plt.subplots()
    plt.plot(-1, 0, 'wo', marker=".", markersize=0)
    plt.plot(0, -1, 'wo', marker=".", markersize=0)
    plt.plot(1, 0, 'wo', marker=".", markersize=0)
    plt.plot(0, 1, 'wo', marker=".", markersize=0)
    plt.plot(0, 0, 'ko', marker="+")
    plt.plot(preds_train[:, 1], preds_train[:, 0], 'ro', label='preds', markersize=2)
    plt.plot(Y_train[:, 1], Y_train[:, 0], 'bo', label='labels', markersize=2)
    ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    fig.suptitle('train preds', fontsize=16)

    fig, ax = plt.subplots()
    plt.plot(-1, 0, 'wo', marker=".", markersize=0)
    plt.plot(0, -1, 'wo', marker=".", markersize=0)
    plt.plot(1, 0, 'wo', marker=".", markersize=0)
    plt.plot(0, 1, 'wo', marker=".", markersize=0)
    plt.plot(0, 0, 'ko', marker="+")
    plt.plot(preds_val[:, 1], preds_val[:, 0], 'ro', label='preds', markersize=2)
    plt.plot(Y_val[:, 1], Y_val[:, 0], 'bo', label='labels', markersize=2)
    ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    fig.suptitle('val preds', fontsize=16)


    fig, ax = plt.subplots()
    plt.plot(-1, 0, 'wo', marker=".", markersize=0)
    plt.plot(0, -1, 'wo', marker=".", markersize=0)
    plt.plot(1, 0, 'wo', marker=".", markersize=0)
    plt.plot(0, 1, 'wo', marker=".", markersize=0)
    plt.plot(0, 0, 'ko', marker="+")
    plt.plot(preds_test[:, 1], preds_test[:, 0], 'ro', label='preds', markersize=2)
    plt.plot(Y_test[:, 1], Y_test[:, 0], 'bo', label='labels', markersize=2)
    ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    fig.suptitle('test preds', fontsize=16)
    plt.show()

    

    # # ***** search hyperparameters *****
    # if(not os.path.isdir('data/logs')):
    #     os.mkdir('data/logs')
    # if(not os.path.isdir('data/logs/flf_nn')):
    #     os.mkdir('data/logs/flf_nn')

    # pbounds = {
    #     'batch_size': (1, 250),
    #     'drop_rate': (0, 0.5),
    #     'lr': (0.01, 0.00001),
    #     'n_epochs': (50, 2500),
    #     'n_neurons': (2, 1028),
    #     'reg_rate': (0, 1)
    # }

    # part = partial(_train, X_train, X_test, Y_train, Y_test, 0)
    # optimizer = BayesianOptimization(f=part, pbounds=pbounds, random_state=1)

    # # load previous logs
    # if(os.path.isfile('data/logs/flf_nn/logs.json')):
    #     load_logs(optimizer, logs=['data/logs/flf_nn/logs.json'])

    # # save logs
    # logger = JSONLogger(path='data/logs/flf_nn/logs.json')
    # optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    # optimizer.maximize(init_points=2, n_iter=3)

if __name__ == '__main__':
    main()