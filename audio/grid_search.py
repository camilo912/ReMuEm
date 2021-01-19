import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from utils import standarize, category_to_id
import os

def get_df():
    cont = 0
    fname = 'data/grid_search_results.csv'

    cs = [1.0, 0.1, 2.0, 0.5, 1.5, 0.05]
    gammas = ['scale', 'auto', 1.0, 0.1, 2.0, 0.5, 1.5, 0.05]
    kernels = ['poly', 'rbf', 'sigmoid']

    if(not os.path.isfile(fname)):
        df = pd.DataFrame(columns=['c','gamma','kernel','degree','acc'])
        for c in cs:
            for gamma in gammas:
                for kernel in kernels:
                    if(kernel == 'poly'):
                        for degree in range(1, 5):
                            df.loc[cont] = [c,gamma,kernel,degree,None]
                            cont += 1
                    else:
                         df.loc[cont] = [c,gamma,kernel,None,None]
                         cont += 1
        df.to_csv(fname, index=False)
    else:
        df = pd.read_csv(fname, header=0)
    return df

def main():
    df = pd.read_csv('data/csvs/data_selected.csv')
    df.columns = np.arange(df.shape[1])
    X = df.loc[:, 4:].values
    # X = df.loc[:, 5:].values
    y = df.loc[:, 1].values
    y, _ = category_to_id(y, 'xxx')

    X = X.astype(np.float64)

    # remove nans
    idxs = []
    for i in range(len(X)):
        if(not np.isnan(X[i,:]).any()):
            idxs.append(i)
    
    X = X[idxs]
    y = y[idxs]

    df = get_df()

    cont = 0
    for c,gamma,kernel,degree,acc in df.values:
        print(cont)
        if(not(acc <= 1)):
            clf = SVC(C=c, gamma=gamma, kernel=kernel, degree=degree)
            acc = np.mean(cross_val_score(clf, X, y, cv=10))
            df.loc[cont, :] = [c,gamma,kernel,degree,acc]
            df.to_csv('data/grid_search_results.csv')
        cont += 1

    # cs = [1.0, 0.1, 2.0, 0.5, 1.5, 0.05]
    # gammas = ['scale', 'auto', 1.0, 0.1, 2.0, 0.5, 1.5, 0.05]
    # kernels = ['poly', 'rbf', 'sigmoid']

    

    # # for c in cs:
    # #     for gamma in gammas:
    # #         for kernel in kernels:
    # #             if(kernel == 'poly'):
    # #                 for degree in range(1, 5):
    # #                     clf = SVC(C=c, gamma=gamma, kernel=kernel, degree=degree)
    # #                     acc = np.mean(cross_val_score(clf, X, y, cv=10))
    # #                     f = open('data/grid_search_results.txt', 'a')
    # #                     f.write('C: '+str(c)+',\tgamma:' + str(gamma)+ ',\tkernel:' + kernel + ',\tdegree:' + str(degree) + ',\tacc:' + str(round(acc,2)) + '\n')
    # #                     f.close()
    # #                     print(c,gamma,kernel,degree,acc)
                        
    # #             else:
    # #                 clf = SVC(C=c, gamma=gamma, kernel=kernel)
    # #                 acc = np.mean(cross_val_score(clf, X, y, cv=10))
    # #                 f = open('data/grid_search_results.txt', 'a')
    # #                 f.write('C: '+str(c)+',\tgamma:' + str(gamma)+ ',\tkernel:' + kernel + ',\t acc:' + str(round(acc,2)) + '\n')
    # #                 f.close()
    # #                 print(c,gamma,kernel,acc)


if __name__ == '__main__':
    main()