import cv2
import numpy as np
import os
from featurizer import Featurizer
import pandas as pd
from model import Model
import pickle
# from predictor import Predictor

def main():
    # dataset = 'enterface'
    # dataset = 'mosei'
    dataset = 'hci'

    if(not os.path.isdir('data')):
        os.mkdir('data')

    if(not os.path.isdir('data/'+dataset)):
        os.mkdir('data/'+dataset)
    
    metadata = pd.read_csv('../metadata_'+dataset+'.csv')
    featurizer = Featurizer()
    data_filename = 'data/'+dataset+'/data_'+str(featurizer.omit)+'.npy'
    labels_filename = 'data/'+dataset+'/labels_'+str(featurizer.omit)+'.npy'

    
    if(not os.path.isfile(data_filename)):
        if(dataset == 'enterface'):
            features, labels = featurizer.run_enterface(metadata)
        elif(dataset == 'mosei'):
            features, labels = featurizer.run_mosei(metadata)
        elif(dataset == 'hci'):
            features, labels = featurizer.run_hci(metadata)
        else:
            raise Exception('Invalid dataset:', dataset)
        np.save(data_filename, features)
        np.save(labels_filename, labels)

    else:
        features = np.load(data_filename, allow_pickle=True)
        labels = np.load(labels_filename, allow_pickle=True)

    selected_filename = 'data/'+dataset+'/data_selected_'+str(featurizer.omit)+'.npy'
    selected_idxs_filename = 'data/'+dataset+'/selected_idxs_'+str(featurizer.omit)+'.npy'

    if(not os.path.isfile(selected_filename)):
        if(dataset == 'enterface' or dataset == 'mosei' or dataset == 'hci'):
            features = featurizer.select_features_continuous(features, labels, dataset)
            raise Exception('Debug')
        else:
            raise Exception('Invalid dataset:', dataset)
        pickle.dump(features, open(selected_filename, 'wb'))
        np.save(selected_idxs_filename, featurizer.idxs)
    else:
        features = np.load(selected_filename, allow_pickle=True)
        idxs = np.load(selected_idxs_filename, allow_pickle=True)

    model = Model()
    model.train_continuous(features, labels, dataset)
    # model.search_hyperparameters(features, labels, dataset)

    # unique = np.unique(labels)
    # dic = {k:i for i,k in enumerate(unique)}
    # dic_inv = {v:k for k,v in dic.items()}
    # predictor = Predictor('data/models/model_0.6441558599472046_100_0.25_10000_0.0001_0.001.h5', featurizer, idxs, dic_inv)
    # cap = cv2.VideoCapture(0)
    # success, img = cap.read()
    # imgs = []
    # for i in range(100):
    #     if(success):
    #         imgs.append(img)
    #     success, img = cap.read()
    # print(len(imgs))
    # predictor.predict(imgs)





if __name__=='__main__':
    main()
