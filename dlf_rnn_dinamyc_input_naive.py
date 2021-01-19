import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l1,l2
from matplotlib import pyplot as plt
from utils import r2_keras
from keras import backend as K

def graph(preds, labels):
    fig, ax = plt.subplots()
    plt.plot(-1, 0, 'wo', marker=".", markersize=0)
    plt.plot(0, -1, 'wo', marker=".", markersize=0)
    plt.plot(1, 0, 'wo', marker=".", markersize=0)
    plt.plot(0, 1, 'wo', marker=".", markersize=0)
    plt.plot(0, 0, 'ko', marker="+")
    plt.plot(preds[:, 1], preds[:, 0], 'ro', label='preds', markersize=2)
    plt.plot(labels[:, 1], labels[:, 0], 'bo', label='labels', markersize=2)
    ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()

    # plt.show()

def custom_loss(y_true, y_pred):
    # y_true = K.cast(y_true, dtype='float32')
    # y_pred = K.cast(y_pred, dtype='float32')
    # loss = K.mean(K.square(y_true - y_pred))
    loss = K.sqrt(K.mean(K.square(y_true - y_pred)))
    # loss += K.square(K.std(y_true)-K.std(y_pred))
    return loss
    # SS_res =  K.sum(K.square(y_true - y_pred)) 
    # SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    # return SS_res/(SS_tot + K.epsilon())
    # return (SS_res/(SS_tot + K.epsilon()))/K.size(y_pred)

def train(X_train, X_test, Y_train, Y_test, verbose):
    # hyper parameters
    batch_size = 32
    drop_rate = 0.0 # 0.375
    lr = 1e-3
    n_epochs = 115
    n_neurons = 64
    reg_rate = 0 # 1e-3 # 1e-1

    model = Sequential()
    # model.add(LSTM(n_neurons, return_sequences=True, activation='relu', bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate), recurrent_regularizer=l2(reg_rate), activity_regularizer=l2(reg_rate), dropout=drop_rate))
    # model.add(Dropout(drop_rate))
    # model.add(LSTM(n_neurons, return_sequences=True, activation='relu', bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate), recurrent_regularizer=l2(reg_rate), activity_regularizer=l2(reg_rate), dropout=drop_rate))
    # model.add(Dropout(drop_rate))
    model.add(LSTM(n_neurons, activation='relu', bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate), recurrent_regularizer=l2(reg_rate), activity_regularizer=l2(reg_rate), dropout=drop_rate))
    # model.add(Dropout(drop_rate))
    model.add(Dense(2, activation='tanh'))
    opt = Adam(learning_rate=lr, clipnorm=0.1)
    model.compile(opt, loss=custom_loss, metrics=[r2_keras])
    history = model.fit(X_train, Y_train, validation_data=[X_test, Y_test], epochs=n_epochs, batch_size=batch_size, verbose=verbose)

    if(verbose):
        plt.subplot(2,1,1)
        plt.plot(history.history['loss'], label='train loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.subplot(2,1,2)
        plt.plot(history.history['r2_keras'], label='train r2')
        plt.plot(history.history['val_r2_keras'], label='val r2')
        plt.legend()
        # plt.show()

    return model

def main():
    # # load data
    # Y_multi_train = np.load('data/mix/preds_train.npy')
    # Y_multi_val = np.load('data/mix/preds_val.npy')
    # Y_multi_test = np.load('data/mix/preds_test.npy')
    # Y_true_train = np.load('data/mix/labels_train.npy')
    # Y_true_val = np.load('data/mix/labels_val.npy')
    # Y_true_test = np.load('data/mix/labels_test.npy')

    Y_multi_train = np.load('data/mix2/preds_train.npy')
    Y_multi_val = np.load('data/mix2/preds_val.npy')
    Y_multi_test = np.load('data/mix2/preds_test.npy')
    Y_true_train = np.load('data/mix2/labels_train.npy')
    Y_true_val = np.load('data/mix2/labels_val.npy')
    Y_true_test = np.load('data/mix2/labels_test.npy')

    # eliminate nans
    # train
    idxs = []
    for i in range(len(Y_multi_train)):
        if(not np.isnan(Y_multi_train[i]).any()):
            idxs.append(i)
    Y_multi_train = Y_multi_train[idxs]
    Y_true_train = Y_true_train[idxs]
    # val
    idxs = []
    for i in range(len(Y_multi_val)):
        if(not np.isnan(Y_multi_val[i]).any()):
            idxs.append(i)
    Y_multi_val = Y_multi_val[idxs]
    Y_true_val = Y_true_val[idxs]
    # test
    idxs = []
    for i in range(len(Y_multi_test)):
        if(not np.isnan(Y_multi_test[i]).any()):
            idxs.append(i)
    Y_multi_test = Y_multi_test[idxs]
    Y_true_test = Y_true_test[idxs]

    print(Y_multi_train.shape)
    print(Y_multi_val.shape)
    print(Y_multi_test.shape)

    model = train(Y_multi_train, Y_multi_val, Y_true_train, Y_true_val, 0)
    preds_train = model.predict(Y_multi_train)
    preds_val = model.predict(Y_multi_val)
    preds_test = model.predict(Y_multi_test)

    print('r2 train:', r2_score(Y_true_train, preds_train))
    print('rmse train:', mean_squared_error(Y_true_train, preds_train)**(1/2))
    print("train std relative error:", (np.linalg.norm(np.std(Y_true_train, axis=0)-np.std(preds_train, axis=0),2)/np.linalg.norm(np.std(Y_true_train, axis=0), 2)))
    print('r2 val:', r2_score(Y_true_val, preds_val))
    print('rmse val:', mean_squared_error(Y_true_val, preds_val)**(1/2))
    print("val std relative error:", (np.linalg.norm(np.std(Y_true_val, axis=0)-np.std(preds_val, axis=0),2)/np.linalg.norm(np.std(Y_true_val, axis=0), 2)))
    print('r2 test:', r2_score(Y_true_test, preds_test))
    print('rmse test:', mean_squared_error(Y_true_test, preds_test)**(1/2))
    print("test std relative error:", (np.linalg.norm(np.std(Y_true_test, axis=0)-np.std(preds_test, axis=0),2)/np.linalg.norm(np.std(Y_true_test, axis=0), 2)))

    graph(preds_train, Y_true_train)
    graph(preds_val, Y_true_val)
    graph(preds_test, Y_true_test)

    errors = Y_true_val - preds_val

    # # aro
    # # reg_aro = RandomForestRegressor(max_depth=3)
    # reg_aro = SVR(C=0.01)
    # reg_aro.fit(Y_multi_val.reshape(-1,6), errors[:,0])
    # preds_errors_aro_val = reg_aro.predict(Y_multi_val.reshape(-1,6)).reshape(-1,1)
    # preds_errors_aro_test = reg_aro.predict(Y_multi_test.reshape(-1,6)).reshape(-1,1)

    # # vale
    # # reg_vale = RandomForestRegressor(max_depth=3)
    # reg_vale = SVR(C=0.01)
    # reg_vale.fit(Y_multi_val.reshape(-1,6), errors[:,1])
    # preds_errors_vale_val = reg_vale.predict(Y_multi_val.reshape(-1,6)).reshape(-1,1)
    # preds_errors_vale_test = reg_vale.predict(Y_multi_test.reshape(-1,6)).reshape(-1,1)

    # preds_errors_val = np.concatenate((preds_errors_aro_val, preds_errors_vale_val), axis=1)
    # preds_errors_test = np.concatenate((preds_errors_aro_test, preds_errors_vale_test), axis=1)

    # clf = RandomForestRegressor(max_depth=2)
    clf = PLSRegression(n_components=1)
    clf.fit(Y_multi_val.reshape(-1,6), errors)
    preds_errors_val = clf.predict(Y_multi_val.reshape(-1,6))
    preds_errors_test = clf.predict(Y_multi_test.reshape(-1,6))

    preds_val = preds_val + preds_errors_val
    preds_test = preds_test + preds_errors_test

    preds_val = np.clip(preds_val, -1, 1)
    preds_test = np.clip(preds_test, -1, 1)

    print('r2 val boosted:', r2_score(Y_true_val, preds_val))
    print('rmse val boosted:', mean_squared_error(Y_true_val, preds_val)**(1/2))
    print("val std relative error:", (np.linalg.norm(np.std(Y_true_val, axis=0)-np.std(preds_val, axis=0),2)/np.linalg.norm(np.std(Y_true_val, axis=0), 2)))
    print('r2 boosted:', r2_score(Y_true_test, preds_test))
    print('rmse boosted:', mean_squared_error(Y_true_test, preds_test)**(1/2))
    print("test std relative error:", (np.linalg.norm(np.std(Y_true_test, axis=0)-np.std(preds_test, axis=0),2)/np.linalg.norm(np.std(Y_true_test, axis=0), 2)))
    graph(preds_val, Y_true_val)
    graph(preds_test, Y_true_test)
    plt.show()

if __name__ == '__main__':
    main()