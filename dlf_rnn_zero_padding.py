import numpy as np
import pandas as pd
import os
from keras import Sequential
from keras.layers import SimpleRNN, Dense, LSTM, Activation, Dropout
from keras.regularizers import l1, l2
from keras.optimizers import Adam
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from functools import partial
from matplotlib import pyplot as plt
from keras.metrics import RootMeanSquaredError
from keras import backend as K
from sklearn.model_selection import train_test_split
from utils import r2_keras
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error

def custom_loss(y_true, y_pred):
    # y_true = K.cast(y_true, dtype='float32')
    # y_pred = K.cast(y_pred, dtype='float32')
    loss = K.mean(K.square(y_true - y_pred))
    # loss += K.square(K.std(y_true)-K.std(y_pred))
    # loss = K.square(K.std(y_true)-K.std(y_pred))
    # loss = K.sqrt(K.sqrt(K.sqrt(K.square(K.var(y_true)-K.var(y_pred)))))
    # loss = K.abs(K.var(y_true)-K.var(y_pred))
    loss += (K.std(y_true)-K.std(y_pred))/10
    return loss
    # SS_res =  K.sum(K.square(y_true - y_pred)) 
    # SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    # loss = SS_res/(SS_tot + K.epsilon())
    # loss += (K.std(y_true)-K.std(y_pred))/K.std(y_true)
    # return loss
    # return (SS_res/(SS_tot + K.epsilon()))/K.size(y_pred)

def create_model(drop_rate, lr, n_neurons, reg_rate):
    model = Sequential()
    # model.add(SimpleRNN(n_neurons))
    model.add(LSTM(n_neurons, input_shape=(None,2), return_sequences=True, bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate)))
    # model.add(LSTM(n_neurons, bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate)))
    model.add(Activation('relu'))
    model.add(Dropout(drop_rate))
    model.add(LSTM(int(n_neurons/2), return_sequences=True, bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate)))
    # model.add(LSTM(int(n_neurons/2), bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate)))
    model.add(Activation('relu'))
    model.add(Dropout(drop_rate))
    model.add(LSTM(int(n_neurons), bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate)))
    model.add(Activation('relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(2))
    model.add(Activation('tanh'))
    opt = Adam(learning_rate=lr)
    model.compile(loss=custom_loss, optimizer=opt, metrics=[RootMeanSquaredError(), r2_keras])

    return model

def train(X_train, X_test, Y_train, Y_test, verbose, batch_size, n_epochs, lr, n_neurons, reg_rate, drop_rate):
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

def _train(X_train, X_test, Y_train, Y_test, verbose, batch_size, n_epochs, lr, n_neurons, reg_rate, drop_rate):
    batch_size = int(batch_size)
    n_epochs = int(n_epochs)
    n_neurons = int(n_neurons)
    model, loss = train(X_train, X_test, Y_train, Y_test, verbose, batch_size, n_epochs, lr, n_neurons, reg_rate, drop_rate)
    return 1-loss

def main():
    # load data
    X_train = np.load('data/mix2/preds_train.npy')
    X_val = np.load('data/mix2/preds_val.npy')
    X_test = np.load('data/mix2/preds_test.npy')
    Y_train = np.load('data/mix2/labels_train.npy')
    Y_val = np.load('data/mix2/labels_val.npy')
    Y_test = np.load('data/mix2/labels_test.npy')

    X_train = np.nan_to_num(X_train)
    X_val = np.nan_to_num(X_val)
    X_test = np.nan_to_num(X_test)

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_val.shape)
    print(Y_test.shape)

    # ***** train *****
    if(not os.path.isdir('data/models')):
        os.mkdir('data/models')
    if(not os.path.isdir('data/models/dlf_rnn_zero_padding')):
        os.mkdir('data/models/dlf_rnn_zero_padding')

    # # arousal
    # # clf = RandomForestRegressor()
    # clf = SVR()
    # clf.fit(X_train.reshape(-1, 6), Y_train[:,0])
    # preds_aro_train = clf.predict(X_train.reshape(-1, 6))
    # preds_aro_val = clf.predict(X_val.reshape(-1, 6))
    # preds_aro_test = clf.predict(X_test.reshape(-1, 6))

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
    # # clf = RandomForestRegressor()
    # clf = SVR()
    # clf.fit(X_train.reshape(-1, 6), Y_train[:,1])
    # preds_vale_train = clf.predict(X_train.reshape(-1, 6))
    # preds_vale_val = clf.predict(X_val.reshape(-1, 6))
    # preds_vale_test = clf.predict(X_test.reshape(-1, 6))

    # preds_vale_train = np.clip(preds_vale_train, -1, 1)
    # preds_vale_val = np.clip(preds_vale_val, -1, 1)
    # preds_vale_test = np.clip(preds_vale_test, -1, 1)

    # print('\n\n')
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

    # # clf = RandomForestRegressor(max_depth=None)
    # clf = PLSRegression(n_components=5)
    # clf.fit(X_train.reshape(-1,6), Y_train)
    # preds_train = clf.predict(X_train.reshape(-1,6))
    # preds_val = clf.predict(X_val.reshape(-1,6))
    # preds_test = clf.predict(X_test.reshape(-1,6))
    
    # hyper parameters
    batch_size = 320
    drop_rate = 0.5
    lr = 1e-3
    n_epochs = 500
    n_neurons = 128
    reg_rate = 0 # 1e-4

    model, loss = train(X_train.reshape(-1,3,2), X_val.reshape(-1,3,2), Y_train, Y_val, 1, batch_size, n_epochs, lr, n_neurons, reg_rate, drop_rate)
    model.save('data/models/dlf_rnn_zero_padding/lstm_rnn_'+str(round(loss,4))+'_'+str(batch_size)+'_'+str(drop_rate)+'_'+str(lr)+'_'+str(n_epochs)+'_'+str(n_neurons)+'_'+str(reg_rate)+'.h5')

    preds_train = model.predict(X_train)
    preds_val = model.predict(X_val)
    preds_test = model.predict(X_test)

    preds_train = np.clip(preds_train, -1, 1)
    preds_val = np.clip(preds_val, -1, 1)
    preds_test = np.clip(preds_test, -1, 1)

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

    np.save('data/outputs/dlf_zero_padding/val_preds.npy', preds_val)
    np.save('data/outputs/dlf_zero_padding/val_data.npy', X_val)
    np.save('data/outputs/dlf_zero_padding/val_errors.npy', errors_val)
    np.save('data/outputs/dlf_zero_padding/val_labels.npy', Y_val)
    np.save('data/outputs/dlf_zero_padding/test_preds.npy', preds_test)
    np.save('data/outputs/dlf_zero_padding/test_data.npy', X_test)
    np.save('data/outputs/dlf_zero_padding/test_errors.npy', errors_test)
    np.save('data/outputs/dlf_zero_padding/test_labels.npy', Y_test)

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
    # if(not os.path.isdir('data/logs/dlf_rnn_zero_padding')):
    #     os.mkdir('data/logs/dlf_rnn_zero_padding')

    # pbounds = {
    #     'batch_size': (1, 250),
    #     'n_epochs': (50, 2500),
    #     'lr': (0.01, 0.00001),
    #     'n_neurons': (2, 1028),
    #     'reg_rate': (0, 1),
    #     'drop_rate': (0,0.5)
    # }

    # part = partial(_train, X_train, X_test, Y_train, Y_test, 0)
    # optimizer = BayesianOptimization(f=part, pbounds=pbounds, random_state=1)

    # # load previous logs
    # if(os.path.isfile('data/logs/dlf_rnn_zero_padding/logs.json')):
    #     load_logs(optimizer, logs=['data/logs/dlf_rnn_zero_padding/logs.json'])

    # # save logs
    # logger = JSONLogger(path='data/logs/dlf_rnn_zero_padding/logs.json')
    # optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    # optimizer.maximize(init_points=2, n_iter=1000)


if __name__ == '__main__':
    main()