from utils import get_useful_triangles, get_cosines, extract_closest_face, emotion_to_id, translate, aro_val_to_emo
import dlib
import wget
import numpy as np
import os
import cv2
from imutils import face_utils
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
import multiprocessing as mp
from functools import partial

detector = dlib.get_frontal_face_detector()
hog_filename = 'data/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(hog_filename)

def parallel(X, i):
    y = X[:, i]
    X_i = np.concatenate((X[:, :i], X[:, i+1:]), axis=1)
    clf = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X_i, y, test_size = 0.3)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

# def parallel_func(calculate_rotation, detector, predictor, gray):
def parallel_func(gray):
    prev_rotation = None
    rotation = calculate_rotation_par(gray, prev_rotation)
    prev_rotation = rotation
    if(rotation is not None):
        gray = cv2.rotate(gray, rotation)
    rects = detector(gray,0)
    if len(rects) >= 1:
        if(len(rects) > 1):
            face = extract_closest_face(rects)
        else:
            face = rects[0]
        height = face.height()
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        landmarks = shape

        face = gray[face.top():face.bottom(), face.left():face.right()]
        return [face, height, landmarks]
    print("face not detected")
    return [None, None, None]
    
def parallel_func_2(data):
    ls, h = data
    features = []
    for c in combinations(range(len(ls)), 2):
        features.append(np.linalg.norm(ls[c[0]]-ls[c[1]], 2)/h)
    return features

def calculate_rotation_par(gray, prev=None):
    rotations = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]
    if(prev is not None):
        tmp = cv2.rotate(gray, prev)
        rects = detector(tmp,0)
        if(len(rects)>0):
            return prev
    
    rects = detector(gray,0)
    cont = 0
    while len(rects) < 1 and cont < len(rotations)-1:
        cont += 1
        tmp = cv2.rotate(gray, rotations[cont])
        rects = detector(tmp,0)
    return rotations[cont]

def wrapper(imgs):
    pool = mp.Pool(mp.cpu_count()-3)
    results = pool.imap(parallel_func, imgs)
    pool.close()

    return results

def wrapper_2(data):
    pool = mp.Pool(mp.cpu_count()-3)
    data = list(pool.imap(parallel_func_2, data))
    pool.close()

    return data

class Featurizer():
    def __init__(self):
        dlib.DLIB_USE_CUDA = True
        self.data_dir = 'data/'
        self.detector = dlib.get_frontal_face_detector()
        self.omit = 2
        hog_filename = self.data_dir + 'shape_predictor_68_face_landmarks.dat'
        if(not os.path.isfile(hog_filename)):
            url = 'https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat?raw=true'
            wget.download(url, hog_filename)
        self.predictor = dlib.shape_predictor(hog_filename)
        self.n_features = 2278


    def calculate_rotation(self, gray, prev=None):
        rotations = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]
        if(prev is not None):
            tmp = cv2.rotate(gray, prev)
            rects = self.detector(tmp,0)
            if(len(rects)>0):
                return prev
        
        rects = self.detector(gray,0)
        cont = 0
        while len(rects) < 1 and cont < len(rotations)-1:
            cont += 1
            tmp = cv2.rotate(gray, rotations[cont])
            rects = self.detector(tmp,0)
        return rotations[cont]

    
    def featurize(self, imgs, sess):
        landmarks = []
        prev_rotation = None
        heights = []
        faces = []

        results = wrapper(imgs)

        for face, height, lands in results:
            if(face is not None):
                faces.append(face)
                heights.append(height)
                landmarks.append(lands)
        np.save('data/images/hci_faces_'+str(sess)+'_'+str(self.omit)+'.npy', faces)
        
        data = wrapper_2(zip(landmarks, heights))

        data = np.array(data)
        
        # resume summarize frames features extracting percentiles 0,25,50,75,100
        if(len(data)>1):
            # features = np.percentile(data, (0,25,50,75,100), 0).ravel()
            # features = data.sum(axis=0)
            features = np.mean(data, axis=0)
        else:
            data = np.array([[None for _ in range(self.n_features)] for _ in range(len(imgs))])
            features = np.array([None for _ in range(self.n_features)])
        np.save('data/features/hci_raw_norm_' + str(sess) + '_' + str(self.omit) + '.npy', data)
        return features

    def run_enterface(self, metadata):
        features = []
        labels = []
        banned = ['subject 6', 'subject 23']

        if(not os.path.isdir('data/features')):
            os.mkdir('data/features')

        print("videos to extract:", len(metadata))
        for i in range(len(metadata)):
            print(i)
            subject, emotion, sentence, path = metadata.loc[i, ['subject','emotion','sentence','path']]
            if(not subject in banned):
                path = '../'+path
                cap = cv2.VideoCapture(path)
                data_fname = 'data/features/dists_mean_norm_' + subject[subject.rindex(' ')+1:] + '_' + emotion + '_' + sentence[-1] + '_' + str(self.omit) + '.npy'

                if(not os.path.isfile(data_fname)):
                    imgs = []
                    success, img = cap.read()

                    while(success):
                        imgs.append(img)
                        # extract only every omit-th frame as in paper: Convolutional MKL Based Multimodal Emotion Recognition and Sentiment Analysis
                        for _ in range(self.omit):
                            cap.read()
                        success, img = cap.read()
                    features_i = self.featurize(imgs)
                    np.save(data_fname, features_i)
                else:
                    features_i = np.load(data_fname, allow_pickle=True)
                features.append(features_i)
                arousal, valence = translate(emotion)
                labels.append([emotion, arousal, valence])
        
        return np.array(features), np.array(labels)
    
    def run_mosei(self, metadata):
        features = []
        labels = []
        banned = []

        if(not os.path.isdir('data/features')):
            os.mkdir('data/features')
        
        print("videos to extract:", len(metadata))
        for i in range(len(metadata)):
            print(i)
            path,vid,clip,emotion,arousal,valence = metadata.loc[i, :]
            if(not str(vid) + '_' + str(clip) in banned):
                path = '../'+path.replace('[modal]', 'Videos').replace('[ext]', '.mp4')
                print(path)
                cap = cv2.VideoCapture(path)
                data_fname = 'data/features/mosei_dists_mean_norm_' + str(vid) + '_' + str(clip) + '_' + str(self.omit) + '.npy'

                if(not os.path.isfile(data_fname)):
                    imgs = []
                    success, img = cap.read()

                    while(success):
                        imgs.append(img)
                        # extract only every omit-th frame as in paper: Convolutional MKL Based Multimodal Emotion Recognition and Sentiment Analysis
                        for _ in range(self.omit):
                            cap.read()
                        success, img = cap.read()
                    features_i = self.featurize(imgs)
                    np.save(data_fname, features_i)
                else:
                    features_i = np.load(data_fname, allow_pickle=True)
                if(len(features)>0):
                    assert len(features_i) == len(features[-1])
                features.append(features_i)
                labels.append([emotion, arousal, valence])
        
        return np.array(features), np.array(labels)
    
    def run_hci(self, metadata):
        data_folder = '../data/hci_tagging/Sessions/'
        features = []
        labels = []
        banned = []

        if(not os.path.isdir('data/features')):
            os.mkdir('data/features')
        if(not os.path.isdir('data/images')):
            os.mkdir('data/images')

        print("videos to extract:", len(metadata))
        for i in range(len(metadata)):
            print(i)
            sess,arousal,valence = metadata.loc[i, :]
            if(not sess in banned):
                emotion = aro_val_to_emo(arousal, valence)
                data_fname = 'data/features/hci_mean_norm_' + str(sess) + '_' + str(self.omit) + '.npy'
                if(not os.path.isfile(data_fname)):
                    avis = [x for x in os.listdir(data_folder+'/'+str(sess)) if '.avi' in x]
                    if(len(avis) == 1):
                        path = '../data/hci_tagging/Sessions/' + str(sess) + '/' + avis[0]
                        print(path)
                        cap = cv2.VideoCapture(path)

                        imgs = []
                        success, img = cap.read()

                        while(success):
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            imgs.append(gray)
                            # extract only every omit-th frame as in paper: Convolutional MKL Based Multimodal Emotion Recognition and Sentiment Analysis
                            for _ in range(self.omit):
                                cap.read()
                            success, img = cap.read()

                        features_i = self.featurize(imgs, sess)
                        np.save(data_fname, features_i)
                    elif(len(avis) > 1):
                        raise Exception('more than 1 avi file for session: ' + str(sess))
                    else:
                        raise Exception('0 avi files for session: ' + str(sess))
                else:
                    features_i = np.load(data_fname, allow_pickle=True)
            else:
                features_i = np.load(data_fname, allow_pickle=True)

            if(len(features)>0):
                assert len(features_i) == len(features[-1])
            features.append(features_i)
            labels.append([emotion, arousal, valence])
        
        return np.array(features), np.array(labels)

    def select_features_continuous(self, X, y, dataset):
        X = X.astype(np.float32)
        df_f = pd.DataFrame(X)
        df_f.columns = np.arange(df_f.shape[1])

        print("features: ", len(df_f.columns))

        epsilon = -1 # 1e-4 # 1e-5
        df_f_tmp = df_f.loc[:,df_f.var()>epsilon]
        tmp = list(df_f_tmp.columns)
        df_f_tmp.columns = np.arange(df_f_tmp.shape[1])

        print("selected after variance filter:", len(tmp))

        X = df_f_tmp.values.astype(np.float32)

        y_tmp = emotion_to_id(y[:,0], dataset)
        # remove nans
        idxs = []
        for i in range(len(X)):
            if(not np.isnan(X[i,:]).any()):
                idxs.append(i)
        
        X = X[idxs]
        y = np.array(y)[idxs]

        X, tmp = self.select_vif(X, tmp)
        df_f = df_f.loc[:, tmp]
        print("selected after VIF filter:", len(tmp))

        y_emo = y[:,0]
        y_emo = emotion_to_id(y_emo, dataset)
        y_aro = y[:, 1].astype(np.float32)
        y_val = y[:, 2].astype(np.float32)

        self.idxs = []

        for mode in ['emo', 'aro', 'val']:
            idxs = []
            if(mode == 'emo'):
                model = RandomForestClassifier(n_estimators=100)
                model.fit(X, y_emo)
                threshold = 1e-3      
            elif(mode == 'aro'):
                model = RandomForestRegressor(n_estimators=100)
                model.fit(X, y_aro)
                threshold = 1e-3
            elif(mode == 'val'):
                model = RandomForestRegressor(n_estimators=100)
                model.fit(X, y_val)
                threshold = 1e-3
            
            importances = model.feature_importances_
            for i in range(len(importances)):
                if(importances[i]>threshold):
                    idxs.append(tmp[i])
            self.idxs.append(np.array(idxs))

            if(mode == 'emo'):
                df_f_emo = df_f.loc[:, importances>threshold]
                print("selected for emotion:", len(df_f_emo.columns))
                X_emo = df_f_emo.values
            elif(mode == 'aro'):
                df_f_aro = df_f.loc[:, importances>threshold]
                print("selected for arousal:", len(df_f_aro.columns))
                X_aro = df_f_aro.values
            elif(mode == 'val'):
                df_f_val = df_f.loc[:, importances>threshold]
                print("selected for valence:", len(df_f_val.columns))
                X_val = df_f_val.values
        X = [X_emo, X_aro, X_val]

        return X

    def select_features_categorical(self, X, y, dataset):
        df = pd.DataFrame(X, columns = np.arange(X.shape[1]))

        print("features: ", len(df.columns))
        epsilon = 1e-4
        df_tmp = df.loc[:,df.var()>epsilon]
        tmp = list(df_tmp.columns) # indexes for knowing which variables are selected
        df_tmp.columns = np.arange(df_tmp.shape[1])

        X = df_tmp.values.astype(np.float32)

        # remove nans
        idxs = []
        for i in range(len(X)):
            if(not np.isnan(X[i,:]).any()):
                idxs.append(i)
        
        X = X[idxs]
        y = y[idxs]
        y = emotion_to_id(y, dataset)

        X, tmp = self.select_vif(X, tmp)
        df = df.loc[:, tmp]
        print("selected after VIF filter:", len(tmp))

        clf = RandomForestClassifier()
        clf.fit(X,y)
        
        self.idxs = []
        # threshold = 5e-3
        threshold = 1e-3
        importances = clf.feature_importances_
        for i in range(len(importances)):
            if(importances[i]>threshold):
                self.idxs.append(tmp[i])
        self.idxs = np.array(self.idxs)

        df = df.loc[:, importances>threshold]
        print("selected:", len(df.columns))

        df.columns = np.arange(df.shape[1])

        return df.values

    def select_vif(self, X, tmp):
        vifs = [np.inf]
        tmp = np.array(tmp)
        while(np.max(vifs) > 10):
            print(len(tmp), np.max(vifs))
            vifs = []

            results = []
            for i in range(len(X.T)):
                results.append(parallel(X, i))
            
            for r in results:
                vifs.append(1/(1-r))
            
            if(len(tmp) > 150):
                if(len(tmp) > 1000):
                    limit = 100
                elif(len(tmp) > 500):
                    limit = 50
                elif(len(tmp) > 250):
                    limit = 25
                else:
                    limit = 5
                
                if(np.max(vifs) > 10):
                    data = [[i, vif] for i,vif in enumerate(vifs)]
                    data = sorted(data, key=lambda x: x[1], reverse=True)
                    idxs = [x[0] for x in data[:limit]]

                    X = np.delete(X, idxs, axis=1)
                    tmp = np.delete(tmp, idxs)

            else:
                print(vifs)
                if(np.max(vifs) > 10):
                    idx = np.argmax(vifs)
                    tmp = np.delete(tmp, idx)
                    X = np.concatenate((X[:, :idx], X[:, idx+1:]), axis=1)
        
        return X, tmp
    
