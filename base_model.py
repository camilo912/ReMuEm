import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import PLSRegression

def main():
    # # preds by modality
    # preds_m_train = np.load('data/mix/preds_train.npy') # m = modality
    # preds_m_val = np.load('data/mix/preds_val.npy')
    # preds_m_test = np.load('data/mix/preds_test.npy') # m = modality
    # labels_b_train = np.load('data/mix/labels_train.npy')
    # labels_b_val = np.load('data/mix/labels_val.npy')
    # labels_b_test = np.load('data/mix/labels_test.npy')

    
    preds_m_train = np.load('data/mix2/preds_train.npy') # m = modality
    preds_m_val = np.load('data/mix2/preds_val.npy')
    preds_m_test = np.load('data/mix2/preds_test.npy') # m = modality
    labels_b_train = np.load('data/mix2/labels_train.npy')
    labels_b_val = np.load('data/mix2/labels_val.npy')
    labels_b_test = np.load('data/mix2/labels_test.npy')

    # rmses = [0.34, 0.55, 0.4]
    rmses = [0.36, 0.54, 0.51]
    inv_rmses = [1-i for i in rmses]
    acum = np.sum(inv_rmses)
    acum_12 = np.sum(inv_rmses[:2])
    acum_13 = inv_rmses[0] + inv_rmses[2]
    acum_23 = np.sum(inv_rmses[1:3])
    weights = [i/acum for i in inv_rmses]
    weights_12 = [i/acum_12 for i in inv_rmses[:2]]
    weights_13 = [i/acum_13 for i in [inv_rmses[0], inv_rmses[2]]]
    weights_23 = [i/acum_23 for i in inv_rmses[1:3]]

    # # create dinamyc samples
    # #train
    # preds_d_train = [] # d = dinamyc
    # labels_train = []
    # for i in range(len(preds_m_train)):
    #     # add trimodal
    #     preds_d_train.append(preds_m_train[i])
    #     labels_train.append(labels_b_train[i])
    #     # add bimodals
    #     preds_d_train.append([preds_m_train[i, 0], preds_m_train[i, 1],  [0.0,0.0]])
    #     labels_train.append(labels_b_train[i])
    #     preds_d_train.append([preds_m_train[i, 0],  [0.0,0.0], preds_m_train[i, 2]])
    #     labels_train.append(labels_b_train[i])
    #     preds_d_train.append([[0.0,0.0], preds_m_train[i, 1],  preds_m_train[i, 2]])
    #     labels_train.append(labels_b_train[i])
    #     # add unimodals
    #     preds_d_train.append([preds_m_train[i, 0],  [0.0,0.0],  [0.0,0.0]])
    #     labels_train.append(labels_b_train[i])
    #     preds_d_train.append([[0.0,0.0], preds_m_train[i, 1],  [0.0,0.0]])
    #     labels_train.append(labels_b_train[i])
    #     preds_d_train.append([[0.0,0.0],  [0.0,0.0], preds_m_train[i, 2]])
    #     labels_train.append(labels_b_train[i])

    # preds_d_train = np.array(preds_d_train, dtype=np.float32)
    # labels_train = np.array(labels_train, dtype=np.float32)

    # #val
    # preds_d_val = [] # d = dinamyc
    # labels_val = []
    # for i in range(len(preds_m_val)):
    #     # add trimodal
    #     preds_d_val.append(preds_m_val[i])
    #     labels_val.append(labels_b_val[i])
    #     # add bimodals
    #     preds_d_val.append([preds_m_val[i, 0], preds_m_val[i, 1],  [0.0,0.0]])
    #     labels_val.append(labels_b_val[i])
    #     preds_d_val.append([preds_m_val[i, 0],  [0.0,0.0], preds_m_val[i, 2]])
    #     labels_val.append(labels_b_val[i])
    #     preds_d_val.append([[0.0,0.0], preds_m_val[i, 1],  preds_m_val[i, 2]])
    #     labels_val.append(labels_b_val[i])
    #     # add unimodals
    #     preds_d_val.append([preds_m_val[i, 0],  [0.0,0.0],  [0.0,0.0]])
    #     labels_val.append(labels_b_val[i])
    #     preds_d_val.append([[0.0,0.0], preds_m_val[i, 1],  [0.0,0.0]])
    #     labels_val.append(labels_b_val[i])
    #     preds_d_val.append([[0.0,0.0],  [0.0,0.0], preds_m_val[i, 2]])
    #     labels_val.append(labels_b_val[i])

    # preds_d_val = np.array(preds_d_val, dtype=np.float32)
    # labels_val = np.array(labels_val, dtype=np.float32)

    # # test
    # preds_d_test = [] # d = dinamyc
    # labels_test = []
    # for i in range(len(preds_m_test)):
    #     # add trimodal
    #     preds_d_test.append(preds_m_test[i])
    #     labels_test.append(labels_b_test[i])
    #     # add bimodals
    #     preds_d_test.append([preds_m_test[i, 0], preds_m_test[i, 1],  [0.0,0.0]])
    #     labels_test.append(labels_b_test[i])
    #     preds_d_test.append([preds_m_test[i, 0],  [0.0,0.0], preds_m_test[i, 2]])
    #     labels_test.append(labels_b_test[i])
    #     preds_d_test.append([[0.0,0.0], preds_m_test[i, 1],  preds_m_test[i, 2]])
    #     labels_test.append(labels_b_test[i])
    #     # add unimodals
    #     preds_d_test.append([preds_m_test[i, 0],  [0.0,0.0],  [0.0,0.0]])
    #     labels_test.append(labels_b_test[i])
    #     preds_d_test.append([[0.0,0.0], preds_m_test[i, 1],  [0.0,0.0]])
    #     labels_test.append(labels_b_test[i])
    #     preds_d_test.append([[0.0,0.0],  [0.0,0.0], preds_m_test[i, 2]])
    #     labels_test.append(labels_b_test[i])

    # preds_d_test = np.array(preds_d_test, dtype=np.float32)
    # labels_test = np.array(labels_test, dtype=np.float32)

    preds_d_train = np.nan_to_num(preds_m_train)
    preds_d_val = np.nan_to_num(preds_m_val)
    preds_d_test = np.nan_to_num(preds_m_test)
    labels_train = labels_b_train
    labels_val = labels_b_val
    labels_test = labels_b_test

    # predict
    # train
    preds_train = []
    for i in range(len(preds_d_train)):
        one = preds_d_train[i,0,0] == 0.0 and preds_d_train[i,0,1] == 0.0
        two = preds_d_train[i,1,0] == 0.0 and preds_d_train[i,1,1] == 0.0
        three = preds_d_train[i,2,0] == 0.0 and preds_d_train[i,2,1] == 0.0
        if(one):
            if(two):
                preds_train.append(preds_d_train[i,2])
            elif(three):
                preds_train.append(preds_d_train[i,1])
            else:
                preds_train.append(preds_d_train[i,1]*weights_23[0] + preds_d_train[i,2]*weights_23[1])
        elif(two):
            if(one):
                preds_train.append(preds_d_train[i,2])
            elif(three):
                preds_train.append(preds_d_train[i,0])
            else:
                preds_train.append(preds_d_train[i,0]*weights_13[0] + preds_d_train[i,2]*weights_13[1])
        elif(three):
            if(one):
                preds_train.append(preds_d_train[i,1])
            elif(two):
                preds_train.append(preds_d_train[i,0])
            else:
                preds_train.append(preds_d_train[i,0]*weights_12[0] + preds_d_train[i,1]*weights_12[1])
        else:
            preds_train.append(preds_d_train[i,0]*weights[0] + preds_d_train[i,1]*weights[1] + preds_d_train[i,2]*weights[2])

    preds_train = np.array(preds_train, dtype=np.float32)

    # val
    preds_val = []
    for i in range(len(preds_d_val)):
        one = preds_d_val[i,0,0] == 0.0 and preds_d_val[i,0,1] == 0.0
        two = preds_d_val[i,1,0] == 0.0 and preds_d_val[i,1,1] == 0.0
        three = preds_d_val[i,2,0] == 0.0 and preds_d_val[i,2,1] == 0.0
        if(one):
            if(two):
                preds_val.append(preds_d_val[i,2])
            elif(three):
                preds_val.append(preds_d_val[i,1])
            else:
                preds_val.append(preds_d_val[i,1]*weights_23[0] + preds_d_val[i,2]*weights_23[1])
        elif(two):
            if(one):
                preds_val.append(preds_d_val[i,2])
            elif(three):
                preds_val.append(preds_d_val[i,0])
            else:
                preds_val.append(preds_d_val[i,0]*weights_13[0] + preds_d_val[i,2]*weights_13[1])
        elif(three):
            if(one):
                preds_val.append(preds_d_val[i,1])
            elif(two):
                preds_val.append(preds_d_val[i,0])
            else:
                preds_val.append(preds_d_val[i,0]*weights_12[0] + preds_d_val[i,1]*weights_12[1])
        else:
            preds_val.append(preds_d_val[i,0]*weights[0] + preds_d_val[i,1]*weights[1] + preds_d_val[i,2]*weights[2])

    preds_val = np.array(preds_val, dtype=np.float32)

    # test
    preds_test = []
    for i in range(len(preds_d_test)):
        one = preds_d_test[i,0,0] == 0.0 and preds_d_test[i,0,1] == 0.0
        two = preds_d_test[i,1,0] == 0.0 and preds_d_test[i,1,1] == 0.0
        three = preds_d_test[i,2,0] == 0.0 and preds_d_test[i,2,1] == 0.0
        if(one):
            if(two):
                preds_test.append(preds_d_test[i,2])
            elif(three):
                preds_test.append(preds_d_test[i,1])
            else:
                preds_test.append(preds_d_test[i,1]*weights_23[0] + preds_d_test[i,2]*weights_23[1])
        elif(two):
            if(one):
                preds_test.append(preds_d_test[i,2])
            elif(three):
                preds_test.append(preds_d_test[i,0])
            else:
                preds_test.append(preds_d_test[i,0]*weights_13[0] + preds_d_test[i,2]*weights_13[1])
        elif(three):
            if(one):
                preds_test.append(preds_d_test[i,1])
            elif(two):
                preds_test.append(preds_d_test[i,0])
            else:
                preds_test.append(preds_d_test[i,0]*weights_12[0] + preds_d_test[i,1]*weights_12[1])
        else:
            preds_test.append(preds_d_test[i,0]*weights[0] + preds_d_test[i,1]*weights[1] + preds_d_test[i,2]*weights[2])

    preds_test = np.array(preds_test, dtype=np.float32)



    med = np.mean(labels_val, axis=0)
    print(med, '//////////////')
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
    
    # random forest predicting errors
    errors_val = labels_val-preds_val

    # preds_aro_val = preds_d_val.reshape(-1,6)
    # preds_vale_val = preds_d_val.reshape(-1,6)
    # preds_aro_test = preds_d_test.reshape(-1,6)
    # preds_vale_test = preds_d_test.reshape(-1,6)
    # errors_aro_val = errors_val[:, 0]
    # errors_vale_val = errors_val[:, 1]

    # # aro
    # # reg_aro = RandomForestRegressor(max_depth=3)
    # reg_aro = SVR()
    # reg_aro.fit(preds_aro_val, errors_aro_val)
    # preds_errors_aro_val = reg_aro.predict(preds_aro_val).reshape(-1,1)
    # preds_errors_aro = reg_aro.predict(preds_aro_test).reshape(-1,1)

    # # vale
    # # reg_vale = RandomForestRegressor(max_depth=3)
    # reg_vale = SVR()
    # reg_vale.fit(preds_vale_val, errors_vale_val)
    # preds_errors_vale_val = reg_vale.predict(preds_vale_val).reshape(-1,1)
    # preds_errors_vale = reg_vale.predict(preds_vale_test).reshape(-1,1)

    # preds_errors_val = np.concatenate((preds_errors_aro_val, preds_errors_vale_val), axis=1)
    # preds_errors = np.concatenate((preds_errors_aro, preds_errors_vale), axis=1)
    # preds_val = preds_val + preds_errors_val
    # preds = preds_test + preds_errors
    # labels = labels_test

    # clf = PLSRegression(n_components=6)
    clf = RandomForestRegressor(n_estimators=500, max_depth=8)
    clf.fit(preds_d_val.reshape(-1,6), errors_val)
    preds_errors_val = clf.predict(preds_d_val.reshape(-1,6))
    preds_errors = clf.predict(preds_d_test.reshape(-1,6))
    preds_val = preds_val + preds_errors_val
    preds = preds_test + preds_errors
    labels = labels_test

    preds_val = np.clip(preds_val, -1, 1)
    preds = np.clip(preds, -1, 1)

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