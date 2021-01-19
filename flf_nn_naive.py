import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from matplotlib import pyplot as plt

def main():
    # load data
    # X_train = np.load('data/mix/data_train.npy')
    # X_val = np.load('data/mix/data_val.npy')
    # X_test = np.load('data/mix/data_test.npy')
    # Y_true_train = np.load('data/mix/labels_train.npy')
    # Y_true_val = np.load('data/mix/labels_val.npy')
    # Y_true_test = np.load('data/mix/labels_test.npy')

    X_train = np.load('data/mix2/data_train.npy')
    X_val = np.load('data/mix2/data_val.npy')
    X_test = np.load('data/mix2/data_test.npy')
    Y_true_train = np.load('data/mix2/labels_train.npy')
    Y_true_val = np.load('data/mix2/labels_val.npy')
    Y_true_test = np.load('data/mix2/labels_test.npy')
    
    # remove nans
    # train
    idxs = []
    for i in range(len(X_train)):
        if(not np.isnan(X_train[i]).any()):
            idxs.append(i)
    X_train = X_train[idxs]
    Y_true_train = Y_true_train[idxs]

    # val
    idxs = []
    for i in range(len(X_val)):
        if(not np.isnan(X_val[i]).any()):
            idxs.append(i)
    X_val = X_val[idxs]
    Y_true_val = Y_true_val[idxs]

    # test
    idxs = []
    for i in range(len(X_test)):
        if(not np.isnan(X_test[i]).any()):
            idxs.append(i)
    X_test = X_test[idxs]
    Y_true_test = Y_true_test[idxs]

    print(X_train.shape)

    # clf = PLSRegression(n_components=5)
    clf = RandomForestRegressor(n_estimators=500)
    clf.fit(X_train, Y_true_train)
    preds_train = clf.predict(X_train)
    preds_val = clf.predict(X_val)
    preds_test = clf.predict(X_test)

    # # aro
    # reg_aro = RandomForestRegressor(n_estimators=500, max_depth=2)
    # reg_aro.fit(X_train.reshape(-1,376), Y_true_train[:,0])
    # preds_aro_train = reg_aro.predict(X_train.reshape(-1,376)).reshape(-1,1)
    # preds_aro_val = reg_aro.predict(X_val.reshape(-1,376)).reshape(-1,1)
    # preds_aro_test = reg_aro.predict(X_test.reshape(-1,376)).reshape(-1,1)

    # # vale
    # reg_vale = RandomForestRegressor(n_estimators=500, max_depth=2)
    # reg_vale.fit(X_train.reshape(-1,376), Y_true_train[:,1])
    # preds_vale_train = reg_vale.predict(X_train.reshape(-1,376)).reshape(-1,1)
    # preds_vale_val = reg_vale.predict(X_val.reshape(-1,376)).reshape(-1,1)
    # preds_vale_test = reg_vale.predict(X_test.reshape(-1,376)).reshape(-1,1)

    # preds_train = np.concatenate((preds_aro_train, preds_vale_train), axis=1)
    # preds_val = np.concatenate((preds_aro_val, preds_vale_val), axis=1)
    # preds_test = np.concatenate((preds_aro_test, preds_vale_test), axis=1)

    print('r2 train:', r2_score(Y_true_train, preds_train))
    print('rmse train:', mean_squared_error(Y_true_train, preds_train)**(1/2))
    print("train std relative error:", (np.linalg.norm(np.std(Y_true_train, axis=0)-np.std(preds_train, axis=0),2)/np.linalg.norm(np.std(Y_true_train, axis=0), 2)))
    print('r2 val:', r2_score(Y_true_val, preds_val))
    print('rmse val:', mean_squared_error(Y_true_val, preds_val)**(1/2))
    print("val std relative error:", (np.linalg.norm(np.std(Y_true_val, axis=0)-np.std(preds_val, axis=0),2)/np.linalg.norm(np.std(Y_true_val, axis=0), 2)))
    print('r2 test:', r2_score(Y_true_test, preds_test))
    print('rmse test:', mean_squared_error(Y_true_test, preds_test)**(1/2))
    print("test std relative error:", (np.linalg.norm(np.std(Y_true_test, axis=0)-np.std(preds_test, axis=0),2)/np.linalg.norm(np.std(Y_true_test, axis=0), 2)))

    fig, ax = plt.subplots()
    plt.plot(-1, 0, 'wo', marker=".", markersize=0)
    plt.plot(0, -1, 'wo', marker=".", markersize=0)
    plt.plot(1, 0, 'wo', marker=".", markersize=0)
    plt.plot(0, 1, 'wo', marker=".", markersize=0)
    plt.plot(0, 0, 'ko', marker="+")
    plt.plot(preds_train[:, 1], preds_train[:, 0], 'ro', label='preds', markersize=2)
    plt.plot(Y_true_train[:, 1], Y_true_train[:, 0], 'bo', label='labels', markersize=2)
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
    plt.plot(Y_true_val[:, 1], Y_true_val[:, 0], 'bo', label='labels', markersize=2)
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
    plt.plot(Y_true_test[:, 1], Y_true_test[:, 0], 'bo', label='labels', markersize=2)
    ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    fig.suptitle('test preds', fontsize=16)

    errors = Y_true_val - preds_val

    # # aro
    # # reg_aro = RandomForestRegressor(max_depth=7)
    # reg_aro = SVR()
    # reg_aro.fit(X_val.reshape(-1,376), errors[:,0])
    # preds_errors_aro_val = reg_aro.predict(X_val.reshape(-1,376)).reshape(-1,1)
    # preds_errors_aro_test = reg_aro.predict(X_test.reshape(-1,376)).reshape(-1,1)

    # # vale
    # # reg_vale = RandomForestRegressor(max_depth=7)
    # reg_vale = SVR()
    # reg_vale.fit(X_val.reshape(-1,376), errors[:,1])
    # preds_errors_vale_val = reg_vale.predict(X_val.reshape(-1,376)).reshape(-1,1)
    # preds_errors_vale_test = reg_vale.predict(X_test.reshape(-1,376)).reshape(-1,1)

    # preds_errors_val = np.concatenate((preds_errors_aro_val, preds_errors_vale_val), axis=1)
    # preds_errors_test = np.concatenate((preds_errors_aro_test, preds_errors_vale_test), axis=1)

    # clf = RandomForestRegressor(n_estimators=500)
    clf = PLSRegression(n_components=8)
    clf.fit(X_val.reshape(-1,376), errors)
    preds_errors_val = clf.predict(X_val.reshape(-1,376))
    preds_errors_test = clf.predict(X_test.reshape(-1,376))

    preds_val = preds_val + preds_errors_val
    preds_test = preds_test + preds_errors_test

    print('r2 val boosted:', r2_score(Y_true_val, preds_val))
    print('rmse val boosted:', mean_squared_error(Y_true_val, preds_val)**(1/2))
    print("val boosted std relative error:", (np.linalg.norm(np.std(Y_true_val, axis=0)-np.std(preds_val, axis=0),2)/np.linalg.norm(np.std(Y_true_val, axis=0), 2)))
    print('r2 boosted:', r2_score(Y_true_test, preds_test))
    print('rmse boosted:', mean_squared_error(Y_true_test, preds_test)**(1/2))
    print("test boosted std relative error:", (np.linalg.norm(np.std(Y_true_test, axis=0)-np.std(preds_test, axis=0),2)/np.linalg.norm(np.std(Y_true_test, axis=0), 2)))

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