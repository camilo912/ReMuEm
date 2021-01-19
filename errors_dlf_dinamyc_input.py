import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.optimizers import Adam
from utils import r2_keras, CustomSequence, create_batches
from keras.metrics import RootMeanSquaredError
from keras.regularizers import l2
from keras import backend as K
import pickle
from matplotlib import pyplot as plt

def predict(X, model):
    preds = []
    for x in X:
        x  = np.array(x).reshape(1,-1,2)
        pred = model.predict(x)
        preds.append(pred)
    return np.array(preds).reshape(-1,2)

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
    # model.add(LSTM(n_neurons, input_shape=(None, 2), return_sequences=True, bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate), activation='relu'))
    # model.add(LSTM(n_neurons, return_sequences=True, bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate), activation='relu'))
    # model.add(Dropout(drop_rate))
    # model.add(LSTM(n_neurons, return_sequences=True, bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate), activation='relu'))
    # model.add(LSTM(n_neurons, return_sequences=True, bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate), activation='relu'))
    # model.add(Dropout(drop_rate))
    # model.add(LSTM(n_neurons, return_sequences=True, bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate), activation='relu'))
    # model.add(LSTM(n_neurons, bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate), activation='relu'))
    # model.add(Dropout(drop_rate))
    model.add(LSTM(n_neurons, input_shape=(None, 2), bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate), activation='relu'))
    model.add(Dropout(drop_rate))
    # model.add(LSTM(int(n_neurons/2), bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate)))
    # model.add(Activation('relu'))
    # model.add(Dropout(drop_rate))
    model.add(Dense(2))
    model.add(Activation('tanh'))
    opt = Adam(learning_rate=lr)
    model.compile(loss=custom_loss, optimizer=opt, metrics=[RootMeanSquaredError(), r2_keras])

    return model

def train(X_train, X_test, Y_train, Y_test, batch_size, drop_rate, lr, n_epochs, n_neurons, reg_rate, verbose):
    model = create_model(drop_rate, lr, n_neurons, reg_rate)
    batches_X_train, batches_X_test, batches_Y_train, batches_Y_test = create_batches(X_train, X_test, Y_train, Y_test, batch_size)

    sequence_train = CustomSequence(batches_X_train, batches_Y_train)
    sequence_test = CustomSequence(batches_X_test, batches_Y_test)

    history = model.fit_generator(sequence_train, validation_data=sequence_test, steps_per_epoch=200, epochs=n_epochs, verbose=verbose, validation_steps=1)

    losses_tr = [x for x in history.history['loss']]
    losses_te = [x for x in history.history['val_loss']]
    rmses_tr = [x for x in history.history['root_mean_squared_error']]
    rmses_te = [x for x in history.history['val_root_mean_squared_error']]
    r2s_tr = [x for x in history.history['r2_keras']]
    r2s_te = [x for x in history.history['val_r2_keras']]
    
    # if(verbose):
    if(True):
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
    X_val = pickle.load(open('data/outputs/dlf_dynamic_input/val_data.pkl', 'rb'))
    X_test = pickle.load(open('data/outputs/dlf_dynamic_input/test_data.pkl', 'rb'))
    errors_val = pickle.load(open('data/outputs/dlf_dynamic_input/val_errors.pkl', 'rb'))
    errors_test = pickle.load(open('data/outputs/dlf_dynamic_input/test_errors.pkl', 'rb'))
    preds_val = pickle.load(open('data/outputs/dlf_dynamic_input/val_preds.pkl', 'rb'))
    preds_test = pickle.load(open('data/outputs/dlf_dynamic_input/test_preds.pkl', 'rb'))
    Y_val = pickle.load(open('data/outputs/dlf_dynamic_input/val_labels.pkl', 'rb'))
    Y_test = pickle.load(open('data/outputs/dlf_dynamic_input/test_labels.pkl', 'rb'))
    print(len(X_val))
    print(len(Y_val))
    print(len(preds_val))

    # X_train, X_test, Y_train, Y_test, idxs_train, idxs_test = train_test_split(X, Y, range(len(Y)), test_size=0.3)
    # preds_test = preds[idxs_test]
    # labels_test = labels[idxs_test]

    # hyper parameters
    batch_size = 500
    drop_rate = 0.5
    lr = 1e-4
    n_epochs = 150
    n_neurons = 32
    reg_rate = 0

    model, loss = train(X_val, X_test, errors_val, errors_test, batch_size, drop_rate, lr, n_epochs, n_neurons, reg_rate, 1)
    model.save('data/models/errors_dlf_dynamic_input/lstm_rnn_'+str(round(loss,4))+ '_' + str(batch_size)+'_'+str(drop_rate)+'_'+str(lr)+'_'+str(n_epochs)+'_'+str(n_neurons)+'_'+str(reg_rate)+'.h5')

    preds_errors_val = predict(X_val, model)
    Y_val_np = np.array(errors_val).reshape(-1, 2)
    preds_errors_test = predict(X_test, model)
    Y_test_np = np.array(errors_test).reshape(-1, 2)

    print('prev r2:', r2_score(Y_test, preds_test))
    print('prev rmse:', mean_squared_error(Y_test, preds_test)**(1/2))
    print("prev std relative error:", (np.linalg.norm(np.std(Y_test, axis=0)-np.std(preds_test, axis=0),2)/np.linalg.norm(np.std(Y_test, axis=0), 2)))

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
    plt.plot(Y_val_np[:, 1], Y_val_np[:, 0], 'bo', label='labels', markersize=1)
    ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()

    fig, ax = plt.subplots()
    plt.plot(-1, 0, 'wo', marker=".", markersize=0)
    plt.plot(0, -1, 'wo', marker=".", markersize=0)
    plt.plot(1, 0, 'wo', marker=".", markersize=0)
    plt.plot(0, 1, 'wo', marker=".", markersize=0)
    plt.plot(0, 0, 'ko', marker="+")
    plt.plot(preds_errors_test[:, 1], preds_errors_test[:, 0], 'ro', label='preds', markersize=2)
    plt.plot(Y_test_np[:, 1], Y_test_np[:, 0], 'bo', label='labels', markersize=1)
    ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()

    preds_val = preds_val + preds_errors_val
    preds_test = preds_test + preds_errors_test
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
    plt.plot(preds_test[:, 1], preds_test[:, 0], 'ro', label='preds', markersize=2)
    plt.plot(Y_test[:, 1], Y_test[:, 0], 'bo', label='labels', markersize=2)
    ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()

    plt.show()



if __name__ == '__main__':
    main()