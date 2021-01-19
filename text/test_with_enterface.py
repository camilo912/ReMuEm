import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score
from utils import emotion_to_id

def main():
    # model = load('data/models/model_55.09.joblib')
    model = load('data/models/model_55.15.joblib')
    # features = np.load('data/data.npy', allow_pickle=True)
    # labels = np.load('data/labels.npy', allow_pickle=True)
    features = np.load('data/data_manually.npy', allow_pickle=True)
    labels = np.load('data/labels_manually.npy', allow_pickle=True)
    idxs = np.load('data/selected_idxs_isear.npy')
    idxs2 = np.load('data/selected_idxs.npy')

    # selected features only
    print("shape before selection:", features.shape)
    print("number of selected for isear dataset:", len(idxs))
    print("number of selected for enterface dataset:", len(idxs2))
    print("number of commom selected:", len(set(idxs).intersection(set(idxs2))))
    features = features[:, idxs].astype(np.float32)
    print("shape after selection:", features.shape)

    # remove nans
    idxs = []
    for i in range(len(features)):
        if(not np.isnan(features[i,:]).any()):
            idxs.append(i)
    
    features = features[idxs]
    labels = labels[idxs]

    print("shape after removing nans:", features.shape)


    # standarize
    means = np.load('data/means_isear.npy', allow_pickle=True)
    stds = np.load('data/stds_isear.npy', allow_pickle=True)
    data = (features - means)/stds

    # convert emotion to id
    labels = emotion_to_id(labels)

    preds = model.predict(data)
    acc = accuracy_score(labels, preds)
    print(acc)


if __name__ == '__main__':
    main()
    