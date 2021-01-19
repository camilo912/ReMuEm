import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib import pyplot as plt

def main():
    # load data
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

    # # aro
    # reg_aro = RandomForestRegressor(n_estimators=500, max_depth=3)
    # # reg_aro = SVR(C=0.5)
    # reg_aro.fit(Y_multi_train.reshape(-1,6), Y_true_train[:,0])
    # preds_aro_train = reg_aro.predict(Y_multi_train.reshape(-1,6)).reshape(-1,1)
    # preds_aro_val = reg_aro.predict(Y_multi_val.reshape(-1,6)).reshape(-1,1)
    # preds_aro_test = reg_aro.predict(Y_multi_test.reshape(-1,6)).reshape(-1,1)

    # # vale
    # reg_vale = RandomForestRegressor(n_estimators=500, max_depth=3)
    # # reg_vale = SVR(C=0.5)
    # reg_vale.fit(Y_multi_train.reshape(-1,6), Y_true_train[:,1])
    # preds_vale_train = reg_vale.predict(Y_multi_train.reshape(-1,6)).reshape(-1,1)
    # preds_vale_val = reg_vale.predict(Y_multi_val.reshape(-1,6)).reshape(-1,1)
    # preds_vale_test = reg_vale.predict(Y_multi_test.reshape(-1,6)).reshape(-1,1)

    # preds_train = np.concatenate((preds_aro_train, preds_vale_train), axis=1)
    # preds_val = np.concatenate((preds_aro_val, preds_vale_val), axis=1)
    # preds_test = np.concatenate((preds_aro_test, preds_vale_test), axis=1)

    # clf = RandomForestRegressor(n_estimators=500, max_depth=4)
    clf = PLSRegression(n_components=4)
    clf.fit(Y_multi_train.reshape(-1,6), Y_true_train)
    preds_train = clf.predict(Y_multi_train.reshape(-1,6))
    preds_val = clf.predict(Y_multi_val.reshape(-1,6))
    preds_test = clf.predict(Y_multi_test.reshape(-1,6))

    preds_train = np.clip(preds_train, -1, 1)
    preds_val = np.clip(preds_val, -1, 1)
    preds_test = np.clip(preds_test, -1, 1)

    print('r2 train:', r2_score(Y_true_train, preds_train))
    print('rmse train:', mean_squared_error(Y_true_train, preds_train)**(1/2))
    print("train std relative error:", (np.linalg.norm(np.std(Y_true_train, axis=0)-np.std(preds_train, axis=0),2)/np.linalg.norm(np.std(Y_true_train, axis=0), 2)))
    print('r2 val:', r2_score(Y_true_val, preds_val))
    print('rmse val:', mean_squared_error(Y_true_val, preds_val)**(1/2))
    print("val std relative error:", (np.linalg.norm(np.std(Y_true_val, axis=0)-np.std(preds_val, axis=0),2)/np.linalg.norm(np.std(Y_true_val, axis=0), 2)))
    print('r2 test:', r2_score(Y_true_test, preds_test))
    print('rmse test:', mean_squared_error(Y_true_test, preds_test)**(1/2))
    print("test std relative error:", (np.linalg.norm(np.std(Y_true_test, axis=0)-np.std(preds_test, axis=0),2)/np.linalg.norm(np.std(Y_true_test, axis=0), 2)))
    
    # fig, ax = plt.subplots()
    # plt.plot(-1, 0, 'wo', marker=".", markersize=0)
    # plt.plot(0, -1, 'wo', marker=".", markersize=0)
    # plt.plot(1, 0, 'wo', marker=".", markersize=0)
    # plt.plot(0, 1, 'wo', marker=".", markersize=0)
    # plt.plot(0, 0, 'ko', marker="+")
    # plt.plot(preds_train[:, 1], preds_train[:, 0], 'ro', label='preds', markersize=2)
    # plt.plot(Y_true_train[:, 1], Y_true_train[:, 0], 'bo', label='labels', markersize=2)
    # ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False))
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.legend()
    # fig.suptitle('train preds', fontsize=16)

    # fig, ax = plt.subplots()
    # plt.plot(-1, 0, 'wo', marker=".", markersize=0)
    # plt.plot(0, -1, 'wo', marker=".", markersize=0)
    # plt.plot(1, 0, 'wo', marker=".", markersize=0)
    # plt.plot(0, 1, 'wo', marker=".", markersize=0)
    # plt.plot(0, 0, 'ko', marker="+")
    # plt.plot(preds_val[:, 1], preds_val[:, 0], 'ro', label='preds', markersize=2)
    # plt.plot(Y_true_val[:, 1], Y_true_val[:, 0], 'bo', label='labels', markersize=2)
    # ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False))
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.legend()
    # fig.suptitle('val preds', fontsize=16)

    # fig, ax = plt.subplots()
    # plt.plot(-1, 0, 'wo', marker=".", markersize=0)
    # plt.plot(0, -1, 'wo', marker=".", markersize=0)
    # plt.plot(1, 0, 'wo', marker=".", markersize=0)
    # plt.plot(0, 1, 'wo', marker=".", markersize=0)
    # plt.plot(0, 0, 'ko', marker="+")
    # plt.plot(preds_test[:, 1], preds_test[:, 0], 'ro', label='preds', markersize=2)
    # plt.plot(Y_true_test[:, 1], Y_true_test[:, 0], 'bo', label='labels', markersize=2)
    # ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False))
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.legend()
    # fig.suptitle('test preds', fontsize=16)

    errors = Y_true_val - preds_val

    # # aro
    # reg_aro = RandomForestRegressor()
    # # reg_aro = SVR()
    # reg_aro.fit(Y_multi_val.reshape(-1,6), errors[:,0])
    # preds_errors_aro_val = reg_aro.predict(Y_multi_val.reshape(-1,6)).reshape(-1,1)
    # preds_errors_aro_test = reg_aro.predict(Y_multi_test.reshape(-1,6)).reshape(-1,1)

    # # vale
    # reg_vale = RandomForestRegressor(max_depth=1)
    # # reg_vale = SVR(C=0.001, epsilon=0.01)
    # reg_vale.fit(Y_multi_val.reshape(-1,6), errors[:,1])
    # preds_errors_vale_val = reg_vale.predict(Y_multi_val.reshape(-1,6)).reshape(-1,1)
    # preds_errors_vale_test = reg_vale.predict(Y_multi_test.reshape(-1,6)).reshape(-1,1)

    # preds_errors_val = np.concatenate((preds_errors_aro_val, preds_errors_vale_val), axis=1)
    # preds_errors_test = np.concatenate((preds_errors_aro_test, preds_errors_vale_test), axis=1)

    # clf = RandomForestRegressor(max_depth=3)
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

    fig, ax = plt.subplots()
    plt.plot(-1, 0, 'wo', marker=".", markersize=0)
    plt.plot(0, -1, 'wo', marker=".", markersize=0)
    plt.plot(1, 0, 'wo', marker=".", markersize=0)
    plt.plot(0, 1, 'wo', marker=".", markersize=0)
    plt.plot(0, 0, 'ko', marker="+")
    plt.plot(preds_val[:, 1], preds_val[:, 0], 'ro', label='preds', markersize=2)
    plt.plot(Y_true_val[:, 1], Y_true_val[:, 0], 'bo', label='labels', markersize=2)
    ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    fig.suptitle('val preds boosted', fontsize=16)

    fig, ax = plt.subplots()
    plt.plot(-1, 0, 'wo', marker=".", markersize=0)
    plt.plot(0, -1, 'wo', marker=".", markersize=0)
    plt.plot(1, 0, 'wo', marker=".", markersize=0)
    plt.plot(0, 1, 'wo', marker=".", markersize=0)
    plt.plot(0, 0, 'ko', marker="+")
    plt.plot(preds_test[:, 1], preds_test[:, 0], 'ro', label='preds', markersize=2)
    plt.plot(Y_true_test[:, 1], Y_true_test[:, 0], 'bo', label='labels', markersize=2)
    ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    fig.suptitle('test preds boosted', fontsize=16)
    plt.show()

if __name__ == '__main__':
    main()