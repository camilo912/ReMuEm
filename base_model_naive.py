import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import PLSRegression

def main():
    # preds by modality
    # preds_m_train = np.load('data/mix/preds_train.npy')
    # preds_m_val = np.load('data/mix/preds_val.npy')
    # preds_m_test = np.load('data/mix/preds_test.npy')
    # labels_train = np.load('data/mix/labels_train.npy')
    # labels_val = np.load('data/mix/labels_val.npy')
    # labels_test = np.load('data/mix/labels_test.npy')


    preds_m_train = np.load('data/mix2/preds_train.npy')
    preds_m_val = np.load('data/mix2/preds_val.npy')
    preds_m_test = np.load('data/mix2/preds_test.npy')
    labels_train = np.load('data/mix2/labels_train.npy')
    labels_val = np.load('data/mix2/labels_val.npy')
    labels_test = np.load('data/mix2/labels_test.npy')

    # eliminate nans
    # train
    idxs = []
    for i in range(len(preds_m_train)):
        if(not np.isnan(preds_m_train[i]).any()):
            idxs.append(i)
    preds_m_train = preds_m_train[idxs]
    labels_train = labels_train[idxs]
    # val
    idxs = []
    for i in range(len(preds_m_val)):
        if(not np.isnan(preds_m_val[i]).any()):
            idxs.append(i)
    preds_m_val = preds_m_val[idxs]
    labels_val = labels_val[idxs]
    # test
    idxs = []
    for i in range(len(preds_m_test)):
        if(not np.isnan(preds_m_test[i]).any()):
            idxs.append(i)
    preds_m_test = preds_m_test[idxs]
    labels_test = labels_test[idxs]

    print(preds_m_train.shape)
    print(preds_m_val.shape)
    print(preds_m_test.shape)

    # rmses = [0.34, 0.55, 0.4]
    rmses = [0.36, 0.54, 0.51]
    inv_rmses = [1-i for i in rmses]
    acum = np.sum(inv_rmses)
    weights = [i/acum for i in inv_rmses]
    preds_train = preds_m_train[:,0]*weights[0] + preds_m_train[:,1]*weights[1] + preds_m_train[:,2]*weights[2]
    preds_val = preds_m_val[:,0]*weights[0] + preds_m_val[:,1]*weights[1] + preds_m_val[:,2]*weights[2]
    # preds_val = (preds_m_val[:,0] + preds_m_val[:,1] + preds_m_val[:,2])/3
    preds_test = preds_m_test[:,0]*weights[0] + preds_m_test[:,1]*weights[1] + preds_m_test[:,2]*weights[2]
    
    med = np.mean(labels_val, axis=0)
    print(med, '*********************')
    print(weights)
    naive = np.array([med for _ in range(len(labels_val))], dtype=np.float32)

    print(preds_val.shape)
    print(labels_val.shape)
    print('R2 naive:', r2_score(labels_val, naive))
    print('RMSE naive:', mean_squared_error(labels_val, naive)**(1/2))
    print("R2 val base model:", r2_score(labels_val, preds_val))
    print("RMSE val base model:", mean_squared_error(labels_val, preds_val)**(1/2))
    print("std val relative error:", (np.linalg.norm(np.std(labels_val, axis=0)-np.std(preds_val, axis=0),2)/np.linalg.norm(np.std(labels_val, axis=0), 2)))
    print("R2 test base model:", r2_score(labels_test, preds_test))
    print("RMSE test base model:", mean_squared_error(labels_test, preds_test)**(1/2))
    print("std test relative error:", (np.linalg.norm(np.std(labels_test, axis=0)-np.std(preds_test, axis=0),2)/np.linalg.norm(np.std(labels_test, axis=0), 2)))

    # fig, ax = plt.subplots()
    # plt.plot(-1, 0, 'wo', marker=".", markersize=0)
    # plt.plot(0, -1, 'wo', marker=".", markersize=0)
    # plt.plot(1, 0, 'wo', marker=".", markersize=0)
    # plt.plot(0, 1, 'wo', marker=".", markersize=0)
    # plt.plot(0, 0, 'ko', marker="+")
    # plt.plot([med[1]], [med[0]], 'go', label='mean', markersize=2)
    # plt.plot(preds_val[:, 1], preds_val[:, 0], 'ro', label='preds', markersize=2)
    # plt.plot(labels_val[:, 1], labels_val[:, 0], 'bo', label='labels', markersize=2)
    # ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False))
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.legend()

    # plt.show()

    # random forest predicting errors
    errors_val = labels_val-preds_val
    # preds_aro_val = preds_m_val.reshape(-1,6)
    # preds_vale_val = preds_m_val.reshape(-1,6)
    # preds_aro_test = preds_m_test.reshape(-1,6)
    # preds_vale_test = preds_m_test.reshape(-1,6)
    # errors_aro_val = errors_val[:, 0]
    # errors_vale_val = errors_val[:, 1]

    # # aro
    # # reg_aro = RandomForestRegressor(max_depth=1)
    # reg_aro = SVR(C=0.01)
    # reg_aro.fit(preds_aro_val, errors_aro_val)
    # preds_errors_aro_val = reg_aro.predict(preds_aro_val).reshape(-1,1)
    # preds_errors_aro = reg_aro.predict(preds_aro_test).reshape(-1,1)

    # # vale
    # # reg_vale = RandomForestRegressor(max_depth=1)
    # reg_vale = SVR(C=0.01)
    # reg_vale.fit(preds_vale_val, errors_vale_val)
    # preds_errors_vale_val = reg_vale.predict(preds_vale_val).reshape(-1,1)
    # preds_errors_vale = reg_vale.predict(preds_vale_test).reshape(-1,1)

    # preds_errors = np.concatenate((preds_errors_aro, preds_errors_vale), axis=1)
    # preds = preds_test + preds_errors
    # labels = labels_test
    # preds_errors_val = np.concatenate((preds_errors_aro_val, preds_errors_vale_val), axis=1)
    # preds_val = preds_val + preds_errors_val

    clf = PLSRegression(n_components=1)
    # clf = RandomForestRegressor(max_depth=1)
    clf.fit(preds_m_val.reshape(-1,6), errors_val)
    preds_errors_val = clf.predict(preds_m_val.reshape(-1,6))
    preds_errors = clf.predict(preds_m_test.reshape(-1,6))
    preds_val = preds_val + preds_errors_val
    preds = preds_test + preds_errors
    labels = labels_test

    print("R2 val boosted base model:", r2_score(labels_val, preds_val))
    print("RMSE val boosted base model:", mean_squared_error(labels_val, preds_val)**(1/2))
    print("val boosted std relative error:", (np.linalg.norm(np.std(labels_val, axis=0)-np.std(preds_val, axis=0),2)/np.linalg.norm(np.std(labels_val, axis=0), 2)))
    print("R2 boosted base model:", r2_score(labels, preds))
    print("RMSE boosted base model:", mean_squared_error(labels, preds)**(1/2))
    print("test boosted std relative error:", (np.linalg.norm(np.std(labels, axis=0)-np.std(preds, axis=0),2)/np.linalg.norm(np.std(labels, axis=0), 2)))

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

    plt.show()



if __name__ == '__main__':
    main()
