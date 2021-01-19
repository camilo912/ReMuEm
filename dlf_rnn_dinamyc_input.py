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
from keras.utils import Sequence
from keras.metrics import RootMeanSquaredError
from keras import backend as K
from utils import r2_keras, CustomSequence, create_batches
import pickle
from sklearn.metrics import r2_score, mean_squared_error

def custom_loss(y_true, y_pred):
    # y_true = K.cast(y_true, dtype='float32')
    # y_pred = K.cast(y_pred, dtype='float32')
    # loss = K.mean(K.square(y_true - y_pred))
    # loss = K.sqrt(K.mean(K.square(y_true - y_pred)))
    # loss += K.square(K.std(y_true)-K.std(y_pred))
    # return loss
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    loss =  SS_res/(SS_tot + K.epsilon())
    loss += (K.std(y_true)-K.std(y_pred))/1.5
    # return (SS_res/(SS_tot + K.epsilon()))/K.size(y_pred)
    return loss

def create_model(drop_rate, lr, n_neurons, reg_rate):
    model = Sequential()
    # model.add(LSTM(n_neurons, input_shape=(None, 2), return_sequences=True, bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate)))
    # model.add(Activation('relu'))
    # model.add(Dropout(drop_rate))
    # model.add(LSTM(int(n_neurons/2), return_sequences=True, bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate)))
    # model.add(Activation('relu'))
    # model.add(Dropout(drop_rate))
    model.add(LSTM(n_neurons, input_shape=(None, 2), bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate), activity_regularizer=l2(reg_rate), recurrent_regularizer=l2(reg_rate), activation='relu'))
    # model.add(LSTM(n_neurons, input_shape=(None, 2), return_sequences=True, bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate), activity_regularizer=l2(reg_rate), recurrent_regularizer=l2(reg_rate), activation='relu'))
    model.add(Dropout(drop_rate))
    # model.add(LSTM(n_neurons, bias_regularizer=l2(reg_rate), return_sequences=True, kernel_regularizer=l2(reg_rate), activity_regularizer=l2(reg_rate), recurrent_regularizer=l2(reg_rate), activation='relu'))
    # model.add(Dropout(drop_rate))
    # model.add(LSTM(n_neurons, input_shape=(None, 2), bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate), activity_regularizer=l2(reg_rate), recurrent_regularizer=l2(reg_rate), activation='relu'))
    # model.add(Dropout(drop_rate))
    model.add(Dense(2))
    model.add(Activation('tanh'))
    opt = Adam(learning_rate=lr)
    model.compile(loss=custom_loss, optimizer=opt, metrics=[RootMeanSquaredError(), r2_keras])

    return model


def train(X_train, X_val, Y_train, Y_val, verbose, batch_size, n_epochs, lr, n_neurons, reg_rate, drop_rate):
    batches_X_train, batches_X_val, batches_Y_train, batches_Y_val = create_batches(X_train, X_val, Y_train, Y_val, batch_size)
    sequence_train = CustomSequence(batches_X_train, batches_Y_train)
    sequence_val = CustomSequence(batches_X_val, batches_Y_val)

    model = create_model(drop_rate, lr, n_neurons, reg_rate)

    # history = model.fit_generator(sequence_train, validation_data=sequence_val, steps_per_epoch=200, epochs=n_epochs, verbose=0, validation_steps=1)
    history = model.fit_generator(sequence_train, validation_data=sequence_val, steps_per_epoch=200, epochs=n_epochs, verbose=verbose, validation_steps=1)

    losses_tr = [x for x in history.history['loss']]
    losses_te = [x for x in history.history['val_loss']]
    rmses_tr = [x for x in history.history['root_mean_squared_error']]
    rmses_te = [x for x in history.history['val_root_mean_squared_error']]
    r2s_tr = [x for x in history.history['r2_keras']]
    r2s_te = [x for x in history.history['val_r2_keras']]
    
    if(verbose):
        plt.subplot(3,1,1)
        plt.plot(losses_tr, label='loss training')
        plt.plot(losses_te, label='loss val')
        plt.legend()
        plt.subplot(3,1,2)
        plt.plot(rmses_tr, label='rmse training')
        plt.plot(rmses_te, label='rmse val')
        plt.legend()
        plt.subplot(3,1,3)
        plt.plot(r2s_tr, label='r2 training')
        plt.plot(r2s_te, label='r2 val')
        plt.legend()
        
    return model, losses_te[-1]

def _train(X_train, X_val, Y_train, Y_val, verbose, batch_size, n_epochs, lr, n_neurons, reg_rate, drop_rate):
    n_epochs = int(n_epochs)
    n_neurons = int(n_neurons)
    model, loss = train(X_train, X_val, Y_train, Y_val, verbose, batch_size, n_epochs, lr, n_neurons, reg_rate, drop_rate)
    return 1-loss

def predict(X, model):
    preds = []
    for x in X:
        x  = np.array(x).reshape(1,-1,2)
        pred = model.predict(x)
        preds.append(pred)
    return np.array(preds).reshape(-1,2)

def main():
    Y_multi_train = np.load('data/mix2/preds_train.npy')
    Y_multi_val = np.load('data/mix2/preds_val.npy')
    Y_multi_test = np.load('data/mix2/preds_test.npy')
    Y_true_train = np.load('data/mix2/labels_train.npy')
    Y_true_val = np.load('data/mix2/labels_val.npy')
    Y_true_test = np.load('data/mix2/labels_test.npy')

    # train
    X_train = []
    Y_train = []
    for X_i, Y_i in zip(Y_multi_train, Y_true_train):
        X_i = np.array([x for x in X_i if not np.isnan(x).any()]).reshape(1,-1,2)
        X_train.append(X_i)
        Y_train.append(Y_i)

    # val
    X_val = []
    Y_val = []
    for X_i, Y_i in zip(Y_multi_val, Y_true_val):
        X_i = np.array([x for x in X_i if not np.isnan(x).any()]).reshape(1,-1,2)
        X_val.append(X_i)
        Y_val.append(Y_i)

    # test
    X_test = []
    Y_test = []
    for X_i, Y_i in zip(Y_multi_test, Y_true_test):
        X_i = np.array([x for x in X_i if not np.isnan(x).any()]).reshape(1,-1,2)
        X_test.append(X_i)
        Y_test.append(Y_i)

    # ***** train *****
    if(not os.path.isdir('data/models')):
        os.mkdir('data/models')
    if(not os.path.isdir('data/models/dlf_rnn_dinamyc_input')):
        os.mkdir('data/models/dlf_rnn_dinamyc_input')
    
    # hyper parameters
    batch_size = 100
    drop_rate = 0.0
    lr = 1e-4
    n_epochs = 200
    n_neurons = 64
    reg_rate = 1e-3 # 1e-2

    model, loss = train(X_train, X_val, Y_train, Y_val, 1, batch_size, n_epochs, lr, n_neurons, reg_rate, drop_rate)
    model.save('data/models/dlf_rnn_dinamyc_input/lstm_rnn_'+str(round(loss,4))+ '_' + str(batch_size)+'_'+str(drop_rate)+'_'+str(lr)+'_'+str(n_epochs)+'_'+str(n_neurons)+'_'+str(reg_rate)+'.h5')

    preds_train = predict(X_train, model)
    Y_train = np.array(Y_train).reshape(-1, 2)
    preds_val = predict(X_val, model)
    Y_val = np.array(Y_val).reshape(-1, 2)
    preds_test = predict(X_test, model)
    Y_test = np.array(Y_test).reshape(-1, 2)

    print("r2 train:", r2_score(Y_train, preds_train))
    print('rmse train:', mean_squared_error(Y_train, preds_train)**(1/2))
    print("std train relative error:", (np.linalg.norm(np.std(Y_train, axis=0)-np.std(preds_train, axis=0),2)/np.linalg.norm(np.std(Y_train, axis=0), 2)))
    print("r2 val:", r2_score(Y_val, preds_val))
    print('rmse val:', mean_squared_error(Y_val, preds_val)**(1/2))
    print("std val relative error:", (np.linalg.norm(np.std(Y_val, axis=0)-np.std(preds_val, axis=0),2)/np.linalg.norm(np.std(Y_val, axis=0), 2)))
    print("r2 test:", r2_score(Y_test, preds_test))
    print('rmse test:', mean_squared_error(Y_test, preds_test)**(1/2))
    print("std test relative error:", (np.linalg.norm(np.std(Y_test, axis=0)-np.std(preds_test, axis=0),2)/np.linalg.norm(np.std(Y_test, axis=0), 2)))

    pickle.dump(X_val, open('data/outputs/dlf_dynamic_input/val_data.pkl', 'wb'))
    pickle.dump(preds_val, open('data/outputs/dlf_dynamic_input/val_preds.pkl', 'wb'))
    pickle.dump(Y_val, open('data/outputs/dlf_dynamic_input/val_labels.pkl', 'wb'))
    pickle.dump(X_test, open('data/outputs/dlf_dynamic_input/test_data.pkl', 'wb'))
    pickle.dump(preds_test, open('data/outputs/dlf_dynamic_input/test_preds.pkl', 'wb'))
    pickle.dump(Y_test, open('data/outputs/dlf_dynamic_input/test_labels.pkl', 'wb'))

    val_errors = Y_val - preds_val
    pickle.dump(val_errors, open('data/outputs/dlf_dynamic_input/val_errors.pkl', 'wb'))
    test_errors = Y_test - preds_test
    pickle.dump(test_errors, open('data/outputs/dlf_dynamic_input/test_errors.pkl', 'wb'))

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
    # if(not os.path.isdir('data/logs/dlf_rnn_dinamyc_input')):
    #     os.mkdir('data/logs/dlf_rnn_dinamyc_input')

    # pbounds = {
    #     'batch_size': (5, 300),
    #     'n_epochs': (5, 1000),
    #     'lr': (0.01, 0.00001),
    #     'n_neurons': (2, 1028),
    #     'reg_rate': (1e-5, 1e-1),
    #     'drop_rate': (0, 0.5)
    # }

    # part = partial(_train, X_train, X_val, Y_train, Y_val, 0)
    # optimizer = BayesianOptimization(f=part, pbounds=pbounds, random_state=1)

    # # load previous logs
    # if(os.path.isfile('data/logs/dlf_rnn_dinamyc_input/logs.json')):
    #     load_logs(optimizer, logs=['data/logs/dlf_rnn_dinamyc_input/logs.json'])

    # # save logs
    # logger = JSONLogger(path='data/logs/dlf_rnn_dinamyc_input/logs.json')
    # optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    # optimizer.maximize(init_points=2, n_iter=1000)


if __name__ == '__main__':
    main()