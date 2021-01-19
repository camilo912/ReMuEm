import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM ,Dropout, Activation
from keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError
from keras.regularizers import l2
from keras import backend as K
from matplotlib import pyplot as plt
from utils import r2_keras

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

def create_model(drop_rate, lr, n_neurons, reg_rate):
    model = Sequential()
    # model.add(LSTM(n_neurons, input_shape=(None, 2), bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate)))
    model.add(LSTM(n_neurons, input_shape=(None, 2), return_sequences=True, bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate)))
    model.add(Activation('relu'))
    model.add(Dropout(drop_rate))
    model.add(LSTM(int(n_neurons/2), bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate)))
    model.add(Activation('relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(2))
    model.add(Activation('tanh'))
    opt = Adam(learning_rate=lr)
    model.compile(loss=custom_loss, optimizer=opt, metrics=[RootMeanSquaredError(), r2_keras])

    return model

def train(X_train, X_test, Y_train, Y_test, batch_size, drop_rate, lr, n_epochs, n_neurons, reg_rate, verbose):
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

def main():
    X_val = np.load('data/outputs/dlf_zero_padding/val_data.npy')
    X_test = np.load('data/outputs/dlf_zero_padding/test_data.npy')
    errors_val = np.load('data/outputs/dlf_zero_padding/val_errors.npy')
    errors_test = np.load('data/outputs/dlf_zero_padding/test_errors.npy')
    preds_val = np.load('data/outputs/dlf_zero_padding/val_preds.npy')
    preds_test = np.load('data/outputs/dlf_zero_padding/test_preds.npy')
    Y_val = np.load('data/outputs/dlf_zero_padding/val_labels.npy')
    Y_test = np.load('data/outputs/dlf_zero_padding/test_labels.npy')

    print('prev r2:', r2_score(Y_test, preds_test))
    print('prev rmse:', mean_squared_error(Y_test, preds_test)**(1/2))
    print("prev std relative error:", (np.linalg.norm(np.std(Y_test, axis=0)-np.std(preds_test, axis=0),2)/np.linalg.norm(np.std(Y_test, axis=0), 2)))

    # arousal
    # clf = RandomForestRegressor()
    clf = SVR()
    clf.fit(X_val.reshape(-1, 6), errors_val[:,0])
    preds_errors_aro_val = clf.predict(X_val.reshape(-1, 6))
    preds_errors_aro_test = clf.predict(X_test.reshape(-1, 6))

    print('\n\n')
    print("r2 errors val aro:", r2_score(errors_val[:,0], preds_errors_aro_val))
    print("rmse errors val aro:", mean_squared_error(errors_val[:,0], preds_errors_aro_val)**(1/2))
    print("std errors val relative error:", (np.std(errors_val[:,0], axis=0)-np.std(preds_errors_aro_val, axis=0))/np.std(errors_val[:,0], axis=0))
    print("r2 errors test aro:", r2_score(errors_test[:,0], preds_errors_aro_test))
    print("rmse errors test aro:", mean_squared_error(errors_test[:,0], preds_errors_aro_test)**(1/2))
    print("std errors test relative error:", (np.std(errors_test[:,0], axis=0)-np.std(preds_errors_aro_test, axis=0))/np.std(errors_test[:,0], axis=0))    

    # valence
    # clf = RandomForestRegressor()
    clf = SVR()
    clf.fit(X_val.reshape(-1, 6), errors_val[:,1])
    preds_errors_vale_val = clf.predict(X_val.reshape(-1, 6))
    preds_errors_vale_test = clf.predict(X_test.reshape(-1, 6))

    print('\n\n')
    print("r2 errors val vale:", r2_score(errors_val[:,1], preds_errors_vale_val))
    print("rmse errors val vale:", mean_squared_error(errors_val[:,1], preds_errors_vale_val)**(1/2))
    print("std errors val relative error:", (np.std(errors_val[:,1], axis=0)-np.std(preds_errors_vale_val, axis=0))/np.std(errors_val[:,1], axis=0))
    print("r2 errors test vale:", r2_score(errors_test[:,1], preds_errors_vale_test))
    print("rmse errors test vale:", mean_squared_error(errors_test[:,1], preds_errors_vale_test)**(1/2))
    print("std errors test relative error:", (np.std(errors_test[:,1], axis=0)-np.std(preds_errors_vale_test, axis=0))/np.std(errors_test[:,1], axis=0))

    preds_errors_val = np.concatenate((preds_errors_aro_val.reshape(-1,1), preds_errors_vale_val.reshape(-1,1)),axis=1)
    preds_errors_test = np.concatenate((preds_errors_aro_test.reshape(-1,1), preds_errors_vale_test.reshape(-1,1)),axis=1)

    # # clf = RandomForestRegressor(max_depth=15)
    # clf = PLSRegression(n_components=6)
    # clf.fit(X_val.reshape(-1,6), errors_val)
    # preds_errors_val = clf.predict(X_val.reshape(-1,6))
    # preds_errors_test = clf.predict(X_test.reshape(-1,6))

    # # NN
    # # hyper parameters
    # batch_size = 32
    # drop_rate = 0.25
    # lr = 1e-4
    # n_epochs = 10000
    # n_neurons = 256
    # reg_rate = 1e-4

    # model, loss = train(X_val, X_test, errors_val, errors_test, batch_size, drop_rate, lr, n_epochs, n_neurons, reg_rate, 1)
    # model.save('data/models/errors_dlf_zero_padding/lstm_rnn_'+str(round(loss,4))+ '_' + str(batch_size)+'_'+str(drop_rate)+'_'+str(lr)+'_'+str(n_epochs)+'_'+str(n_neurons)+'_'+str(reg_rate)+'.h5')

    # preds_errors_val = model.predict(X_val)
    # preds_errors_test = model.predict(X_test)

    print('\n\n')
    print('r2 errors val:', r2_score(errors_val, preds_errors_val))
    print('rmse errors val:', mean_squared_error(errors_val, preds_errors_val)**(1/2))
    print('std relative error errors val:', (np.linalg.norm(np.std(errors_val, axis=0)-np.std(preds_errors_val, axis=0), 2))/np.linalg.norm(np.std(errors_val, axis=0),2))
    print('r2 errors test:', r2_score(errors_test, preds_errors_test))
    print('rmse errors test:', mean_squared_error(errors_test, preds_errors_test)**(1/2))
    print('std relative error errors test:', (np.linalg.norm(np.std(errors_test, axis=0)-np.std(preds_errors_test, axis=0), 2))/np.linalg.norm(np.std(errors_test, axis=0),2))
    

    fig, ax = plt.subplots()
    plt.plot(-1, 0, 'wo', marker=".", markersize=0)
    plt.plot(0, -1, 'wo', marker=".", markersize=0)
    plt.plot(1, 0, 'wo', marker=".", markersize=0)
    plt.plot(0, 1, 'wo', marker=".", markersize=0)
    plt.plot(0, 0, 'ko', marker="+")
    plt.plot(preds_errors_val[:, 1], preds_errors_val[:, 0], 'ro', label='preds', markersize=2)
    plt.plot(errors_val[:, 1], errors_val[:, 0], 'bo', label='labels', markersize=2)
    ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    fig.suptitle('val errors preds', fontsize=16)

    fig, ax = plt.subplots()
    plt.plot(-1, 0, 'wo', marker=".", markersize=0)
    plt.plot(0, -1, 'wo', marker=".", markersize=0)
    plt.plot(1, 0, 'wo', marker=".", markersize=0)
    plt.plot(0, 1, 'wo', marker=".", markersize=0)
    plt.plot(0, 0, 'ko', marker="+")
    plt.plot(preds_errors_test[:, 1], preds_errors_test[:, 0], 'ro', label='preds', markersize=2)
    plt.plot(errors_test[:, 1], errors_test[:, 0], 'bo', label='labels', markersize=2)
    ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    fig.suptitle('test errors preds', fontsize=16)
    
    preds_val = preds_val + preds_errors_val
    preds_test = preds_test + preds_errors_test

    preds_val = np.clip(preds_val, -1, 1)
    preds_test = np.clip(preds_test, -1, 1)

    print('\n\n')
    print("r2 val:", r2_score(Y_val, preds_val))
    print("rmse val:", mean_squared_error(Y_val, preds_val)**(1/2))
    print("std val relative error:", (np.linalg.norm(np.std(Y_val, axis=0)-np.std(preds_val, axis=0),2)/np.linalg.norm(np.std(Y_val, axis=0), 2)))
    print("r2 test vale:", r2_score(Y_test, preds_test))
    print("rmse test vale:", mean_squared_error(Y_test, preds_test)**(1/2))
    print("std test relative error:", (np.linalg.norm(np.std(Y_test, axis=0)-np.std(preds_test, axis=0),2)/np.linalg.norm(np.std(Y_test, axis=0), 2)))

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



if __name__ == '__main__':
    main()
