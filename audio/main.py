import numpy as np
import pandas as pd
from featurizer import Featurizer
import os
from model import Model
import pickle

def main():
    # dataset = 'enterface'
    dataset = 'iemocap'
    # dataset = 'mosei'
    # dataset = 'hci'

    metadata = pd.read_csv('../metadata_'+dataset+'.csv')
    featurizer = Featurizer()

    if(not os.path.isdir('data')):
        os.mkdir('data')

    if(not os.path.isdir('data/'+dataset)):
        os.mkdir('data/'+dataset)

    if(not os.path.isfile('data/'+dataset+'/data.npy')):
        if(dataset == 'enterface'):
            features, labels = featurizer.run_enterface(metadata)
        elif(dataset == 'iemocap'):
            features, labels = featurizer.run_iemocap(metadata)
        elif(dataset == 'mosei'):
            features, labels = featurizer.run_mosei(metadata)
        elif(dataset == 'hci'):
            features, labels = featurizer.run_hci(metadata)
        else:
            raise Exception('invalid dataset: ' + dataset)
        np.save('data/'+dataset+'/data.npy', features)
        np.save('data/'+dataset+'/labels.npy', labels)
    else:
        features = np.load('data/'+dataset+'/data.npy', allow_pickle=True)
        labels = np.load('data/'+dataset+'/labels.npy', allow_pickle=True)
    
    selected_idxs_filename = 'data/'+dataset+'/selected_idxs.npy'
    if(not os.path.isfile('data/'+dataset+'/data_selected.npy')):
        if(dataset=='enterface'):
            features = featurizer.select_features_categorical(features, labels, dataset)
            np.save('data/'+dataset+'/data_selected.npy', features)
        elif(dataset=='iemocap' or dataset=='mosei' or dataset=='hci'):
            features = featurizer.select_features_continuous(features, labels, dataset)
            pickle.dump(features, open('data/'+dataset+'/data_selected.npy', 'wb'))
        np.save(selected_idxs_filename, featurizer.idxs)
    else:
        features = np.load('data/'+dataset+'/data_selected.npy', allow_pickle=True)

    model = Model()
    if(dataset == 'enterface'):
        model.train_categorical(features, labels, dataset)
    elif(dataset == 'iemocap' or dataset=='mosei' or dataset=='hci'):
        model.train_continuous(features, labels, dataset)
        # model.search_hyperparameters(features, labels, dataset)
    else:
        raise Exception('invalid dataset: ' + dataset)

if(__name__ == '__main__'):
    main()