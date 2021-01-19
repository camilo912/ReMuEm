import numpy as np
import pandas as pd
from featurizer import Featurizer
import os
from model import Model
import pickle

def main():
    # dataset = 'isear'
    # dataset = 'enterface'
    # dataset = 'enterface_manually'
    # dataset = 'iemocap'
    # dataset = 'mosei'
    dataset = 'fb'
    featurizer = Featurizer()

    if(not os.path.isdir('data')):
        os.mkdir('data')

    if(not os.path.isdir('data/'+dataset)):
        os.mkdir('data/'+dataset)

    if(dataset == 'isear'):
        metadata = pd.read_csv('data/csvs/isear_data.csv')
        if(not os.path.isfile('data/'+dataset+'/data.npy')):
            features, labels = featurizer.run_isear(metadata)
            np.save('data/'+dataset+'/data.npy', features)
            np.save('data/'+dataset+'/labels.npy', labels)
        else:
            features = np.load('data/'+dataset+'/data.npy', allow_pickle=True)
            labels = np.load('data/'+dataset+'/labels.npy', allow_pickle=True)

        selected_idxs_filename = 'data/'+dataset+'/selected_idxs.npy'
        if(not os.path.isfile('data/'+dataset+'/data_selected.npy')):
            features = featurizer.select_features(features, labels)
            np.save('data/'+dataset+'/data_selected.npy', features)
            np.save(selected_idxs_filename, featurizer.idxs)
        else:
            features = np.load('data/'+dataset+'/data_selected.npy', allow_pickle=True)
        
        model = Model()
        model.train_categorical(features, labels, dataset)
    elif(dataset == 'enterface'):
        metadata = pd.read_csv('../metadata_'+dataset+'.csv')
        if(not os.path.isfile('data/'+dataset+'/data.npy')):
            features, labels = featurizer.run_enterface(metadata)
            np.save('data/'+dataset+'/data.npy', features)
            np.save('data/'+dataset+'/labels.npy', labels)
        else:
            features = np.load('data/'+dataset+'/data.npy', allow_pickle=True)
            labels = np.load('data/'+dataset+'/labels.npy', allow_pickle=True)

        selected_idxs_filename = 'data/'+dataset+'/selected_idxs.npy'
        if(not os.path.isfile('data/'+dataset+'/data_selected.npy')):
            features = featurizer.select_features(features, labels)
            np.save('data/'+dataset+'/data_selected.npy', features)
            np.save(selected_idxs_filename, featurizer.idxs)
    elif(dataset == 'enterface_manually'):
        metadata = pd.read_csv('data/csvs/manually.csv')
        if(not os.path.isfile('data/'+dataset+'/data.npy')):
            features, labels = featurizer.run_enterface(metadata)
            np.save('data/'+dataset+'/data.npy', features)
            np.save('data/'+dataset+'/labels.npy', labels)
        else:
            features = np.load('data/'+dataset+'/data.npy', allow_pickle=True)
            labels = np.load('data/'+dataset+'/labels.npy', allow_pickle=True)

        selected_idxs_filename = 'data/'+dataset+'/selected_idxs.npy'
        if(not os.path.isfile('data/'+dataset+'/data_selected.npy')):
            features = featurizer.select_features(features, labels)
            np.save('data/'+dataset+'/data_selected.npy', features)
            np.save(selected_idxs_filename, featurizer.idxs)
        else:
            features = np.load('data/'+dataset+'/data_selected.npy', allow_pickle=True)
    elif(dataset=='iemocap'):
        metadata = pd.read_csv('../metadata_'+dataset+'.csv')
        if(not os.path.isfile('data/'+dataset+'/data.npy')):
            features, labels = featurizer.run_iemocap(metadata)
            np.save('data/'+dataset+'/data.npy', features)
            np.save('data/'+dataset+'/labels.npy', labels)
        else:
            features = np.load('data/'+dataset+'/data.npy', allow_pickle=True)
            labels = np.load('data/'+dataset+'/labels.npy', allow_pickle=True)

        selected_idxs_filename = 'data/'+dataset+'/selected_idxs.npy'
        if(not os.path.isfile('data/'+dataset+'/data_selected.npy')):
            features = featurizer.select_features_continuous(features, labels, dataset)
            pickle.dump(features, open('data/'+dataset+'/data_selected.npy','wb'))
            np.save(selected_idxs_filename, featurizer.idxs)
        else:
            features = np.load('data/'+dataset+'/data_selected.npy', allow_pickle=True)
        
        model = Model()
        # model.train_continuous(features, labels, dataset)
        model.search_hyperparameters(features, labels, dataset)
    elif(dataset == 'mosei'):
        metadata = pd.read_csv('../metadata_'+dataset+'.csv')
        if(not os.path.isfile('data/'+dataset+'/data.npy')):
            features, labels = featurizer.run_mosei(metadata)
            np.save('data/'+dataset+'/data.npy', features)
            np.save('data/'+dataset+'/labels.npy', labels)
        else:
            features = np.load('data/'+dataset+'/data.npy', allow_pickle=True)
            labels = np.load('data/'+dataset+'/labels.npy', allow_pickle=True)

        selected_idxs_filename = 'data/'+dataset+'/selected_idxs.npy'
        if(not os.path.isfile('data/'+dataset+'/data_selected.npy')):
            features = featurizer.select_features_continuous(features, labels, dataset)
            pickle.dump(features, open('data/'+dataset+'/data_selected.npy','wb'))
            np.save(selected_idxs_filename, featurizer.idxs)
        else:
            features = np.load('data/'+dataset+'/data_selected.npy', allow_pickle=True)
        
        model = Model()
        model.train_continuous(features, labels, dataset)
        # model.search_hyperparameters(features, labels, dataset)
    elif(dataset == 'fb'):
        metadata = pd.read_csv('../metadata_'+dataset+'.csv')
        if(not os.path.isfile('data/'+dataset+'/data.npy')):
            features, labels = featurizer.run_fb(metadata)
            np.save('data/'+dataset+'/data.npy', features)
            np.save('data/'+dataset+'/labels.npy', labels)
        else:
            features = np.load('data/'+dataset+'/data.npy', allow_pickle=True)
            labels = np.load('data/'+dataset+'/labels.npy', allow_pickle=True)

        selected_idxs_filename = 'data/'+dataset+'/selected_idxs.npy'
        if(not os.path.isfile('data/'+dataset+'/data_selected.npy')):
            features = featurizer.select_features_continuous(features, labels, dataset)
            pickle.dump(features, open('data/'+dataset+'/data_selected.npy','wb'))
            np.save(selected_idxs_filename, featurizer.idxs)
        else:
            features = np.load('data/'+dataset+'/data_selected.npy', allow_pickle=True)
        
        model = Model()
        model.train_continuous(features, labels, dataset)
        # model.search_hyperparameters(features, labels, dataset)
    else:
        raise Exception('Invalid dataset:', datasset)



if(__name__ == '__main__'):
    main()