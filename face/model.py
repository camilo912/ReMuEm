import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression, PLSCanonical, PLSSVD, CCA
from sklearn.metrics import mean_squared_error as mse, f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from utils import standarize, emotion_to_id, fuse_arousal_valence_data, aro_val_to_emo
from scipy.stats import pearsonr
import multiprocessing as mp
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load
from keras.metrics import RootMeanSquaredError
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from functools import partial
import os
from tqdm import tqdm
import pickle

def train_hyperparameters(X_fuse,y_fuse,batch_size,drop_rate,epochs,lr,reg_rate):
    n = 3
    rmse = 0

    batch_size = int(batch_size)
    epochs = int(epochs)

    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X_fuse, y_fuse, test_size=0.3)
        model = Sequential()
        model.add(Dense(512, bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate)))
        model.add(Activation('relu'))
        model.add(Dropout(drop_rate))
        model.add(Dense(256, bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate)))
        model.add(Activation('relu'))
        model.add(Dropout(drop_rate))
        model.add(Dense(64, bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate)))
        model.add(Activation('relu'))
        model.add(Dropout(drop_rate))
        model.add(Dense(2, bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate)))
        opt = Adam(learning_rate=lr)
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=[RootMeanSquaredError()])
        history = model.fit(X_train, y_train,batch_size=batch_size, epochs=epochs, validation_data=[X_test, y_test], verbose=0)
        rmse += history.history['val_root_mean_squared_error'][-1]

    return 1-(rmse/n)

def parallel_func(ent):
    sess, aro, val, selected_idxs = ent 
    data = np.load('data/features/hci_raw_norm_'+str(sess)+'_2.npy')
    if(len(data)> 0):
        X_emo_i = data[:, selected_idxs[0]]
        X_aro_i = data[:, selected_idxs[1]]
        X_val_i = data[:, selected_idxs[2]]
        y_emo_i = aro_val_to_emo(aro,val)
        y_aro_i = (aro-1)/4-1
        y_val_i = val
        return [X_emo_i, X_aro_i, X_val_i, y_emo_i, y_aro_i, y_val_i]
    return [None, None, None, None, None, None]

class Model():
    def train_continuous(self, X, y, dataset):
        if(dataset == 'mosei'):
            print(X[0].shape)
            print(X[1].shape)
            print(X[2].shape)

            wall = int(len(y)*0.25)

            X_emo_train = X[0][:wall].astype(np.float32)
            X_emo_test = X[0][wall:wall*2].astype(np.float32)
            X_aro_train = X[1][:wall].astype(np.float32)
            X_aro_test = X[1][wall:wall*2].astype(np.float32)
            X_val_train = X[2][:wall].astype(np.float32)
            X_val_test = X[2][wall:wall*2].astype(np.float32)
            X_preds_aro = X[1][:].astype(np.float32)
            X_preds_val = X[2][:].astype(np.float32)

            y_emo_train = y[:wall, 0]
            y_emo_test = y[wall:wall*2, 0]
            y_aro_train = y[:wall, 1].astype(np.float32)
            y_aro_test = y[wall:wall*2, 1].astype(np.float32)
            y_val_train = y[:wall, 2].astype(np.float32)
            y_val_test = y[wall:wall*2, 2].astype(np.float32)
            y_preds_aro = y[:, 1].astype(np.float32)
            y_preds_val = y[:, 2].astype(np.float32)

            assert len(X_emo_train) == wall
            assert len(X_emo_test) == wall
            assert len(X_aro_train) == wall
            assert len(X_aro_test) == wall
            assert len(X_val_train) == wall
            assert len(X_val_test) == wall
            assert len(y_emo_train) == wall
            assert len(y_emo_test) == wall
            assert len(y_aro_train) == wall
            assert len(y_aro_test) == wall
            assert len(y_val_train) == wall
            assert len(y_val_test) == wall

                
            # remove nans
            idxs_emo = []
            idxs_aro_val = []
            for i in range(len(X_emo_train)):
                if(not np.isnan(X_emo_train[i]).any()):
                    idxs_emo.append(i)
                if(not np.isnan(X_aro_train[i]).any() and not np.isnan(X_val_train[i]).any()):
                    idxs_aro_val.append(i)
            X_emo_train = X_emo_train[idxs_emo]
            y_emo_train = y_emo_train[idxs_emo]
            X_aro_train = X_aro_train[idxs_aro_val]
            y_aro_train = y_aro_train[idxs_aro_val]
            X_val_train = X_val_train[idxs_aro_val]
            y_val_train = y_val_train[idxs_aro_val]

            idxs_emo = []
            idxs_aro_val = []
            for i in range(len(X_emo_test)):
                if(not np.isnan(X_emo_test[i]).any()):
                    idxs_emo.append(i)
                if(not np.isnan(X_aro_test[i]).any() and not np.isnan(X_val_test[i]).any()):
                    idxs_aro_val.append(i)
            X_emo_test = X_emo_test[idxs_emo]
            y_emo_test = y_emo_test[idxs_emo]
            X_aro_test = X_aro_test[idxs_aro_val]
            y_aro_test = y_aro_test[idxs_aro_val]
            X_val_test = X_val_test[idxs_aro_val]
            y_val_test = y_val_test[idxs_aro_val]

            y_emo_train = emotion_to_id(y_emo_train, dataset)
            y_emo_test = emotion_to_id(y_emo_test, dataset)

            # standarize for emotion
            X_emo_train, means, stds = standarize(X_emo_train)
            np.save('data/' + dataset + '/means_emo.npy', means)
            np.save('data/' + dataset + '/stds_emo.npy', stds)
            X_emo_test = (X_emo_test-means)/stds

            # standarize for arousal
            X_aro_train, means, stds = standarize(X_aro_train)
            np.save('data/' + dataset + '/means_aro.npy', means)
            np.save('data/' + dataset + '/stds_aro.npy', stds)
            X_aro_test = (X_aro_test-means)/stds
            X_preds_aro = (X_preds_aro-means)/stds

            # standarize for valence
            X_val_train, means, stds = standarize(X_val_train)
            np.save('data/' + dataset + '/means_val.npy', means)
            np.save('data/' + dataset + '/stds_val.npy', stds)
            X_val_test = (X_val_test-means)/stds
            X_preds_val = (X_preds_val-means)/stds

            # emotion
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(X_emo_train, y_emo_train)
            acc = clf.score(X_emo_test, y_emo_test)
            print('accuracy for emotions:', acc)

            # arousal
            clf = RandomForestRegressor(n_estimators=100)
            clf.fit(X_aro_train, y_aro_train)
            r2_aro = clf.score(X_aro_test, y_aro_test)
            print('R2 for arousal:', r2_aro)
            preds_aro = clf.predict(X_aro_test)
            rmse_aro = (mean_squared_error(y_aro_test, preds_aro))**(1/2)
            print('rmse for arousal:', rmse_aro)


            # valence
            clf = RandomForestRegressor(n_estimators=100)
            clf.fit(X_val_train, y_val_train)
            r2_val = clf.score(X_val_test, y_val_test)
            print('R2 for valence:', r2_val)
            preds_val = clf.predict(X_val_test)
            rmse_val = (mean_squared_error(y_val_test, preds_val))**(1/2)
            print('rmse for valence:', rmse_val)

            # NN

            # fuse arousal and valence data
            X_train = fuse_arousal_valence_data(X_aro_train, X_val_train)
            X_test = fuse_arousal_valence_data(X_aro_test, X_val_test)
            y_train = np.concatenate((y_aro_train.reshape(-1,1), y_val_train.reshape(-1,1)), axis=1)
            y_test = np.concatenate((y_aro_test.reshape(-1,1), y_val_test.reshape(-1,1)), axis=1)

            print(X_train.shape)
            print(X_test.shape)
            print(y_train.shape)
            print(y_test.shape)

            #hyper parameters
            batch_size = 212
            drop_rate = 0.33289238211597166
            epochs = 1913
            lr = 0.01
            reg_rate = 0.01

            model = Sequential()
            model.add(Dense(512, bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate)))
            model.add(Activation('relu'))
            model.add(Dropout(drop_rate))
            model.add(Dense(256, bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate)))
            model.add(Activation('relu'))
            model.add(Dropout(drop_rate))
            model.add(Dense(64, bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate)))
            model.add(Activation('relu'))
            model.add(Dropout(drop_rate))
            model.add(Dense(2, bias_regularizer=l2(reg_rate), kernel_regularizer=l2(reg_rate)))
            opt = Adam(learning_rate=lr)
            model.compile(loss='mean_squared_error', optimizer=opt, metrics=[RootMeanSquaredError()])
            history = model.fit(X_train, y_train,batch_size=batch_size, epochs=epochs, validation_data=[X_test, y_test], verbose=0)
            rmse = round(history.history['val_root_mean_squared_error'][-1], 2)
            print(rmse)

            if(not os.path.isdir('data/'+dataset+'/models/')):
                os.mkdir('data/'+dataset+'/models/')
            model.save('data/'+dataset+'/models/model_'+str(rmse)+'_'+str(batch_size)+'_'+str(round(drop_rate, 2))+'_'+str(epochs)+'_'+str(lr)+'_'+str(reg_rate)+'.h5')

            plt.subplot(2,1,1)
            plt.plot(history.history['loss'], label='train loss')
            plt.plot(history.history['val_loss'], label='validation loss')
            plt.subplot(2,1,2)
            plt.legend()
            plt.plot(history.history['root_mean_squared_error'], label='train rmse')
            plt.plot(history.history['val_root_mean_squared_error'], label='validation rmse')
            plt.legend()
            plt.show()

            print(X_preds_aro.shape)
            print(X_preds_val.shape)
            print(y_preds_aro.shape)
            print(y_preds_val.shape)

            idxs = []
            nan_idxs = []

            y_preds_all = np.concatenate((y_preds_aro.reshape(-1,1), y_preds_val.reshape(-1,1)), axis=1)

            for i in range(len(X_preds_aro)):
                if(not np.isnan(X_preds_aro[i]).any() and not np.isnan(X_preds_val[i]).any()):
                    idxs.append(i)
                else:
                    nan_idxs.append(i)
            
            X_preds_aro = X_preds_aro[idxs]
            X_preds_val = X_preds_val[idxs]
            y_preds_aro = y_preds_aro[idxs]
            y_preds_val = y_preds_val[idxs]

            X_preds = fuse_arousal_valence_data(X_preds_aro, X_preds_val)
            y_preds = np.concatenate((y_preds_aro.reshape(-1,1), y_preds_val.reshape(-1,1)), axis=1)

            print(X_preds.shape)
            print(y_preds.shape)
            print(len(idxs))
            print(len(nan_idxs))

            preds = model.predict(X_preds)
            rmse_preds = (mean_squared_error(y_preds, preds))**(1/2)
            print(rmse_preds)
            print(preds.shape)

            for idx in nan_idxs:
                preds = np.insert(preds, idx, np.array([np.nan, np.nan]), 0)
            
            print(preds.shape)
            np.save('data/'+dataset+'/preds.npy', preds)
        elif(dataset=='hci'):
            models_dir = 'data/'+dataset+'/models/'
            if(not os.path.isdir(models_dir)):
                os.mkdir(models_dir)

            X = fuse_arousal_valence_data(X[1].astype(np.float32), X[2].astype(np.float32))
            Y = y[:,1:3].astype(np.float32)

            # remove nans
            idxs = []
            for i in range(len(X)):
                if(not np.isnan(X[i]).any()):
                    idxs.append(i)
            X = X[idxs]
            # Y = (Y[idxs]-1)/4.0-1.0
            Y = Y[idxs]
            X, x_means, x_stds = standarize(X)
            Y, y_means, y_stds = standarize(Y)

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.7)

            # eliminate high density points
            print("before eliminating high density points:",len(X_train))
            X_tr = []
            Y_tr = []
            for i in range(len(X_train)):
                band = True
                y_i = Y_train[i]
                cont = 0
                for p in Y_tr:
                    if(np.linalg.norm(p-y_i,2) < 0.05):
                        cont +=1
                        if(cont == 2):
                            band = False
                            break
                if(band):
                    X_tr.append(X_train[i])
                    Y_tr.append(y_i)
            X_train = np.array(X_tr)
            Y_train = np.array(Y_tr)
            print("after eliminating high density points:",len(X_train))

            # # fix bias
            # for _ in range(50):
            #     maxi = -0.1
            #     idx = -1
            #     aro_m = np.mean(Y_train[:,0])
            #     val_m = np.mean(Y_train[:,1])
            #     w_aro = aro_m/np.abs(aro_m+val_m+np.finfo(np.float32).eps)
            #     w_val = val_m/np.abs(aro_m+val_m+np.finfo(np.float32).eps)
            #     for i in range(len(X_train)):
            #         p = Y_train[i,0]*w_aro + Y_train[i,1]*w_val
            #         if(p > maxi):
            #             maxi = p
            #             idx = i
            #     X_train = np.append(X_train[:idx], X_train[idx+1:], axis=0)
            #     Y_train = np.append(Y_train[:idx], Y_train[idx+1:], axis=0)
            
            print(X_train.shape)
            clf = PLSRegression(n_components=7)
            # clf = PLSCanonical(n_components=5)
            # clf = CCA(n_components=7)
            # clf = RandomForestRegressor(n_estimators=500, max_depth=6)
            # clf = RandomForestRegressor(n_estimators=500, max_features=8)
            # clf = RandomForestRegressor(n_estimators=500)
            clf.fit(X_train, Y_train)
            # preds_train = np.clip(clf.predict(X_train),-1,1)
            # preds_test = np.clip(clf.predict(X_test),-1,1)
            preds_train = clf.predict(X_train)
            preds_test = clf.predict(X_test)

            preds_train = preds_train*y_stds + y_means
            preds_test = preds_test*y_stds + y_means
            Y_train = Y_train*y_stds + y_means
            Y_test = Y_test*y_stds + y_means

            preds_train = (preds_train-1)/4-1
            preds_test = (preds_test-1)/4-1
            preds_train = np.clip(preds_train, -1, 1)
            preds_test = np.clip(preds_test, -1, 1)
            Y_train = (Y_train-1)/4-1
            Y_test = (Y_test-1)/4-1

            diff_means_train = np.linalg.norm(np.mean(Y_train, axis=0)-np.mean(preds_train, axis=0),2)/np.linalg.norm(np.mean(Y_train, axis=0),2)
            diff_stds_train = np.linalg.norm(np.std(Y_train, axis=0)-np.std(preds_train, axis=0),2)/np.linalg.norm(np.std(Y_train, axis=0),2)
            diff_means_test = np.linalg.norm(np.mean(Y_test, axis=0)-np.mean(preds_test, axis=0),2)/np.linalg.norm(np.mean(Y_test, axis=0),2)
            diff_stds_test = np.linalg.norm(np.std(Y_test, axis=0)-np.std(preds_test, axis=0),2)/np.linalg.norm(np.std(Y_test, axis=0),2)

            print("R2 train:", r2_score(Y_train, preds_train))
            print("RMSE train:", mean_squared_error(Y_train, preds_train)**(1/2))
            print("diff means train:", diff_means_train)
            print("diff stds  train:", diff_stds_train)
            print("R2 test:", r2_score(Y_test, preds_test))
            print("RMSE test:", mean_squared_error(Y_test, preds_test)**(1/2))
            print("diff means test:", diff_means_test)
            print("diff stds test:", diff_stds_test)

            clf = PLSRegression(n_components=7)
            clf.fit(np.append(X_train, X_test, axis=0), np.append(Y_train, Y_test, axis=0))
            dump(clf, models_dir + 'model.joblib')

            # save data
            np.save('data/'+dataset+'/test_data.npy', X_test)
            np.save('data/'+dataset+'/test_preds.npy', preds_test)
            np.save('data/'+dataset+'/test_labels.npy', Y_test)

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
            plt.plot(preds_test[:, 1], preds_test[:, 0], 'ro', label='preds', markersize=2)
            plt.plot(Y_test[:, 1], Y_test[:, 0], 'bo', label='labels', markersize=2)
            ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False))
            plt.gca().set_aspect('equal', adjustable='box')
            plt.legend()
            fig.suptitle('test preds', fontsize=16)
            
            plt.show()
        elif(dataset == 'hcia'):
            models_dir = 'data/'+dataset+'/models/'
            if(not os.path.isdir(models_dir)):
                os.mkdir(models_dir)

            X_emo = X[0].astype(np.float32)
            X_aro = X[1].astype(np.float32)
            X_val = X[2].astype(np.float32)
            y_emo = y[:, 0]
            y_aro = y[:, 1].astype(np.float32)
            y_val = y[:, 2].astype(np.float32)

            # remove nans
            idxs_emo = []
            idxs = []
            for i in range(len(X_emo)):
                if(not np.isnan(X_emo[i]).any()):
                    idxs_emo.append(i)
                if(not np.isnan(X_aro[i]).any() and not np.isnan(X_val[i]).any()):
                    idxs.append(i)
            X_emo = X_emo[idxs_emo]
            y_emo = y_emo[idxs_emo]
            X_aro = X_aro[idxs]
            y_aro = (y_aro[idxs]-1)/4-1
            X_val = X_val[idxs]
            y_val = (y_val[idxs]-1)/4-1

            print(pd.Series(y_aro).describe())
            print(pd.Series(y_val).describe())

            print(X_emo.shape)
            print(X_aro.shape)
            print(X_val.shape)
            print(y_emo.shape)
            print(y_aro.shape)
            print(y_val.shape)

            y_emo = emotion_to_id(y_emo, dataset)

            X_emo_train, X_emo_test, y_emo_train, y_emo_test = train_test_split(X_emo, y_emo, test_size=0.7)
            idxs = np.arange(len(idxs))
            X_aro_train, X_aro_test, y_aro_train, y_aro_test, idxs_train, idxs_test = train_test_split(X_aro, y_aro, idxs, test_size=0.7)
            #X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(X_val, y_val, test_size=0.3)
            X_val_train = X_val[idxs_train]
            X_val_test = X_val[idxs_test]
            y_val_train = y_val[idxs_train]
            y_val_test = y_val[idxs_test]

            # emotion
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(X_emo_train, y_emo_train)
            acc = clf.score(X_emo_test, y_emo_test)
            print('accuracy for emotions:', acc)

            # arousal
            clf = RandomForestRegressor(n_estimators=500)
            clf.fit(X_aro_train, y_aro_train)
            r2_aro = clf.score(X_aro_test, y_aro_test)
            print('R2 train arousal:', clf.score(X_aro_train, y_aro_train))
            print('R2 for arousal:', r2_aro)
            preds_aro_train = clf.predict(X_aro_train)
            preds_aro = clf.predict(X_aro_test)
            rmse_aro = (mean_squared_error(y_aro_test, preds_aro))**(1/2)
            print('rmse train arousal:', mean_squared_error(y_aro_train, preds_aro_train)**(1/2))
            print('rmse for arousal:', rmse_aro)
            model_aro = RandomForestRegressor(n_estimators=500)
            model_aro.fit(X_aro, y_aro)
            dump(model_aro, models_dir + 'model_aro.joblib')

            # valence
            clf = RandomForestRegressor(n_estimators=500)
            clf.fit(X_val_train, y_val_train)
            r2_val = clf.score(X_val_test, y_val_test)
            print('R2 train valence:', clf.score(X_val_train, y_val_train))
            print('R2 for valence:', r2_val)
            preds_val_train = clf.predict(X_val_train)
            preds_val = clf.predict(X_val_test)
            rmse_val = (mean_squared_error(y_val_test, preds_val))**(1/2)
            print('rmse train valence:', mean_squared_error(y_val_train, preds_val_train)**(1/2))
            print('rmse for valence:', rmse_val)
            model_val = RandomForestRegressor(n_estimators=500)
            model_val.fit(X_val, y_val)
            dump(model_val, models_dir + 'model_val.joblib')

            X_fuse = fuse_arousal_valence_data(X_aro_test, X_val_test)
            np.save('data/'+dataset+'/test_data.npy', X_fuse)
            preds = np.append(preds_aro.reshape(-1, 1), preds_val.reshape(-1, 1), axis=1)
            np.save('data/'+dataset+'/test_preds.npy', preds)
            labels = np.append(y_aro_test.reshape(-1, 1), y_val_test.reshape(-1, 1), axis=1)
            np.save('data/'+dataset+'/test_labels.npy', labels)
            preds_train = np.append(preds_aro_train.reshape(-1, 1), preds_val_train.reshape(-1, 1), axis=1)
            labels_train = np.append(y_aro_train.reshape(-1, 1), y_val_train.reshape(-1, 1), axis=1)
            print('r2 train final:', r2_score(labels_train, preds_train))
            print('rmse train final:', mean_squared_error(labels_train, preds_train)**(1/2))
            print("r2 final:", r2_score(labels, preds))
            print("rmse final:", mean_squared_error(labels, preds)**(1/2))

            fig, ax = plt.subplots()
            plt.plot(-1, 0, 'wo', marker=".", markersize=0)
            plt.plot(0, -1, 'wo', marker=".", markersize=0)
            plt.plot(1, 0, 'wo', marker=".", markersize=0)
            plt.plot(0, 1, 'wo', marker=".", markersize=0)
            plt.plot(0, 0, 'ko', marker="+")
            plt.plot(preds_train[:, 1], preds_train[:, 0], 'ro', label='preds', markersize=2)
            plt.plot(labels_train[:, 1], labels_train[:, 0], 'bo', label='labels', markersize=2)
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
            plt.plot(preds[:, 1], preds[:, 0], 'ro', label='preds', markersize=2)
            plt.plot(labels[:, 1], labels[:, 0], 'bo', label='labels', markersize=2)
            ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False))
            plt.gca().set_aspect('equal', adjustable='box')
            plt.legend()
            fig.suptitle('test preds', fontsize=16)
            
            plt.show()
            # return 0
            

            #######################
            # evaluate with mosei #
            #######################
            selected_idxs = np.load('data/'+dataset+'/selected_idxs_2.npy', allow_pickle=True)
            X_mosei = np.load('data/intensity_mosei/data_2.npy', allow_pickle=True)
            X_mosei_aro = X_mosei[:, selected_idxs[1]].astype(np.float32)
            X_mosei_val = X_mosei[:, selected_idxs[2]].astype(np.float32)
            y_mosei = np.load('data/intensity_mosei/labels_2.npy')
            y_mosei_aro = y_mosei[:, 1].astype(np.float32)
            y_mosei_val = y_mosei[:, 2].astype(np.float32)

            # X_mosei_aro_train, X_mosei_aro_test, y_mosei_aro_train, y_mosei_aro_test = train_test_split(X_mosei_aro, y_mosei_aro, test_size=0.3)
            # X_mosei_val_train, X_mosei_val_test, y_mosei_val_train, y_mosei_val_test = train_test_split(X_mosei_val, y_mosei_val, test_size=0.3)

            # remove nans
            idxs = []
            for i in range(len(X_mosei)):
                if(not np.isnan(X_mosei_aro[i]).any() and not np.isnan(X_mosei_val[i]).any()):
                    idxs.append(i)
            
            print(len(idxs))
            X_mosei_aro = X_mosei_aro[idxs]
            X_mosei_val = X_mosei_val[idxs]
            y_mosei_aro = y_mosei_aro[idxs]
            y_mosei_val = y_mosei_val[idxs]

            print(pd.Series(y_mosei_aro).describe())
            print(pd.Series(y_mosei_val).describe())

            preds_aro = model_aro.predict(X_mosei_aro)
            rmse_aro = (mean_squared_error(y_mosei_aro, preds_aro))**(1/2)
            print('R2 arousal mosei:', model_aro.score(X_mosei_aro, y_mosei_aro))
            print('RMSE arousal mosei:', rmse_aro)

            preds_val = model_val.predict(X_mosei_val)
            rmse_val = (mean_squared_error(y_mosei_val, preds_val))**(1/2)
            print('R2 valence mosei:', model_val.score(X_mosei_val, y_mosei_val))
            print('RMSE valence mosei:', rmse_val)

            preds = []
            preds_labels = []
            cont = 0
            for i in range(len(y_mosei)):
                if(i in idxs):
                    preds.append([preds_aro[cont], preds_val[cont]])
                    preds_labels.append([y_mosei[cont, 1], y_mosei[cont, 2]])
                    cont += 1
                else:
                    preds.append([np.nan, np.nan])
                    preds_labels.append([np.nan, np.nan])
            
            assert cont == len(idxs)
            np.save('data/'+dataset+'/preds.npy', preds)
            np.save('data/'+dataset+'/preds_labels.npy', preds_labels)

        else:
            raise Exception('Invalid dataset: ' + dataset + ', at model')

    def search_hyperparameters(self, X, y, dataset):
        print(X[0].shape)
        print(X[1].shape)
        print(X[2].shape)

        # remove nans
        idxs = []
        if(X[0].ndim == 2):
            for i in range(len(X[0])):
                if(not np.isnan(X[0][i,:].astype(np.float32)).any()):
                    idxs.append(i)
        else:
            for i in range(len(X)):
                if(not np.isnan(X[i,:].astype(np.float32)).any()):
                    idxs.append(i)

        y = y[idxs, :]

        if(X[0].ndim == 2):
            X_emo = X[0][idxs,:].astype(np.float32)
            X_aro = X[1][idxs,:].astype(np.float32)
            X_val = X[2][idxs,:].astype(np.float32)
        elif(X[0].ndim == 1):
            X_emo = X.astype(np.float32)
            X_aro = X.astype(np.float32)
            X_val = X.astype(np.float32)

        y_emo = y[:,0]
        y_aro = y[:,1].astype(np.float32)
        y_val = y[:,2].astype(np.float32)

        y_emo = emotion_to_id(y_emo, dataset)

        # standarize for emotion
        X_emo, means, stds = standarize(X_emo)
        np.save('data/'+dataset+'/means_emo.npy', means)
        np.save('data/'+dataset+'/stds_emo.npy', stds)

        # standarize for arousal
        X_aro, means, stds = standarize(X_aro)
        np.save('data/'+dataset+'/means_aro.npy', means)
        np.save('data/'+dataset+'/stds_aro.npy', stds)

        # standarize for valence
        X_val, means, stds = standarize(X_val)
        np.save('data/'+dataset+'/means_val.npy', means)
        np.save('data/'+dataset+'/stds_val.npy', stds)

        # fuse arousal and valence data
        X_fuse = fuse_arousal_valence_data(X_aro, X_val)
        y_fuse = np.concatenate((y_aro.reshape(-1,1), y_val.reshape(-1,1)), axis=1)

        pbounds = {
            'batch_size': (25, 500),
            'drop_rate': (0.0, 0.7),
            'epochs': (50, 2500),
            'lr': (0.01, 0.00001),
            'reg_rate': (0.01, 0.00001)
        }

        part = partial(train_hyperparameters, X_fuse, y_fuse)
        optimizer = BayesianOptimization(f=part, pbounds=pbounds, random_state=1)

        # load previous logs
        if(os.path.isfile('data/'+dataset+'/logs.json')):
            load_logs(optimizer, logs=['data/'+dataset+'/logs.json'])

        # save logs
        logger = JSONLogger(path='data/'+dataset+'/logs.json')
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        optimizer.maximize(init_points=2, n_iter=3000)

        