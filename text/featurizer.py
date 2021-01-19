import pandas as pd
from senticnet5 import create_senticnet
from nltk.tokenize import word_tokenize
import string
import re
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from utils import emotion_to_id, aro_val_to_emo
from sklearn.model_selection import train_test_split
import os

def preprocess(sent, valids):
    sent = sent.lower()
    table = str.maketrans('', '', string.punctuation.replace("-", ""))
    sent = sent.translate(table)
    words = [x for x in word_tokenize(sent) if x in valids]
    return words

def parallel(X, i):
    y = X[:, i]
    X_i = np.concatenate((X[:, :i], X[:, i+1:]), axis=1)
    clf = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X_i, y, test_size = 0.3)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

class Featurizer():
    def __init__(self):
        create_senticnet()
        #self.df_sentic = pd.read_csv('data/senticnet5.csv', index_col=0)
        #self.df_sentic = self.df_sentic[['pleasantness_value', 'attention_value', 'sensitivity_value', 'aptitude_value', 'primary_mood', 'secondary_mood', 'polarity_value']]
        self.df_sentic = pd.read_csv('data/mixed.csv', index_col=0, header=0)
        self.df_sentic = self.df_sentic[['pleasantness_value', 'attention_value', 'sensitivity_value', 'aptitude_value', 'primary_mood', 'secondary_mood', 'polarity_value'] + [str(i+1)  for i in range(100)]]
        self.n_features = 541
    
    def run_enterface(self, metadata):
        data_folder = '../data/enterface/enterface database/'
        stopwords = [line.rstrip('\n') for line in open('data/stop_words.txt')]
        valids = list(self.df_sentic.index)
        features = []
        labels = []
        banned = ['subject 6', 'subject 23']
        numeric = ['pleasantness_value', 'attention_value', 'sensitivity_value', 'aptitude_value', 'polarity_value']
        numeric = numeric + [str(i+1)  for i in range(100)]
        for subject, emotion, sentence, path in metadata.values:
            if(not subject in banned):
                fname = '../' + path.replace('.avi', '.txt')
                f = open(fname, 'r')
                line = ' '.join([x.strip() for x in f.readlines()])
                words = preprocess(line, valids)
                n = len(words)
                if(len(words) > 0):
                    data = self.df_sentic.loc[words, :]
                    features_i = np.percentile(data[numeric], (0,25,50,75,100), 0).ravel()
                    # features_i = data[numeric].sum().values.ravel()
                    tag_to_id = {'#joy':0, '#sadness':1, '#interest':2, '#surprise':3, '#anger':4, '#fear':5, '#admiration':6, '#disgust':7}
                    tags = np.zeros((8), dtype=np.float32)
                    for k,v in data['primary_mood'].value_counts().items():
                        tags[tag_to_id[k]] = v/n
                    features_i = np.append(features_i, tags, axis=0)

                    tags = np.zeros((8), dtype=np.float32)
                    for k,v in data['secondary_mood'].value_counts().items():
                        tags[tag_to_id[k]] = v/n
                    features_i = np.append(features_i, tags, axis=0)
                else:
                    features_i = np.array([None for _ in range(self.n_features)])
                features.append(features_i)
                labels.append(emotion)
                f.close()
        
        features = np.array(features)
        labels = np.array(labels)
        return features, labels
    
    def run_isear(self, metadata):
        features = []
        labels = []
        valids = list(self.df_sentic.index)
        stopwords = [line.rstrip('\n') for line in open('data/stop_words.txt')]
        numeric = ['pleasantness_value', 'attention_value', 'sensitivity_value', 'aptitude_value', 'polarity_value']
        numeric = numeric + [str(i+1)  for i in range(100)]
        for emotion, text in metadata.values:
            words = preprocess(text, valids)
            n = len(words)
            if(len(words) > 0):
                data = self.df_sentic.loc[words, :]
                features_i = np.percentile(data[numeric], (0,25,50,75,100), 0).ravel()
                # features_i = data[numeric].sum().values.ravel()
                # features_i = data[numeric].mean().values.ravel()
                tag_to_id = {'#joy':0, '#sadness':1, '#interest':2, '#surprise':3, '#anger':4, '#fear':5, '#admiration':6, '#disgust':7}
                tags = np.zeros((8), dtype=np.float32)
                for k,v in data['primary_mood'].value_counts().items():
                    tags[tag_to_id[k]] = v/n
                features_i = np.append(features_i, tags, axis=0)

                tags = np.zeros((8), dtype=np.float32)
                for k,v in data['secondary_mood'].value_counts().items():
                    tags[tag_to_id[k]] = v/n
                features_i = np.append(features_i, tags, axis=0)
            else:
                features_i = np.array([None for _ in range(self.n_features)])
            features.append(features_i)
            labels.append(emotion)
        
        features = np.array(features)
        labels = np.array(labels)
        return features, labels
    
    def run_iemocap(self, metadata):
        data_folder = '../data/IEMOCAP_full_release/'
        features = []
        labels = []
        prev_fname = None
        valids = list(self.df_sentic.index)
        stopwords = [line.rstrip('\n') for line in open('data/stop_words.txt')]
        numeric = ['pleasantness_value', 'attention_value', 'sensitivity_value', 'aptitude_value', 'polarity_value']
        numeric = numeric + [str(i+1)  for i in range(100)]
        for i in range(len(metadata)):
            sess, fname, utt_name, emotion, arousal, valence = metadata.loc[i, ['session','filename','uterance_name', 'emotion', 'arousal', 'valence']]

            if(prev_fname is None or prev_fname != fname):
                path = data_folder + sess + '/dialog/transcriptions/' + fname + '.txt'
                f = open(path, 'r')
                prev_fname = fname
                lines = f.readlines()
                utterances = {l[l.index('[')-5:l.index('[')-1]:l[l.index(':')+2:].strip() for l in lines if '[' in l and ':' in l}
            
            line = utterances[utt_name]
            words = preprocess(line, valids)
            n = len(words)
            if(len(words) > 0):
                data = self.df_sentic.loc[words, :]
                features_i = np.percentile(data[numeric], (0,25,50,75,100), 0).ravel()
                # features_i = data[numeric].sum().values.ravel()
                # features_i = data[numeric].mean().values.ravel()
                tag_to_id = {'#joy':0, '#sadness':1, '#interest':2, '#surprise':3, '#anger':4, '#fear':5, '#admiration':6, '#disgust':7}
                tags = np.zeros((8), dtype=np.float32)
                for k,v in data['primary_mood'].value_counts().items():
                    tags[tag_to_id[k]] = v/n
                features_i = np.append(features_i, tags, axis=0)

                tags = np.zeros((8), dtype=np.float32)
                for k,v in data['secondary_mood'].value_counts().items():
                    tags[tag_to_id[k]] = v/n
                features_i = np.append(features_i, tags, axis=0)
            else:
                features_i = np.array([None for _ in range(self.n_features)])
            features.append(features_i)
            labels.append([emotion, arousal, valence])
        
        features = np.array(features)
        labels = np.array(labels)
        return features, labels
    
    def run_mosei(self, metadata):
        data_folder = '../data/mosei/Raw/Transcript/Segmented/Combined/'
        df = pd.DataFrame()
        features= []
        labels = []
        valids = list(self.df_sentic.index)
        stopwords = [line.rstrip('\n') for line in open('data/stop_words.txt')]
        numeric = ['pleasantness_value', 'attention_value', 'sensitivity_value', 'aptitude_value', 'polarity_value']
        numeric = numeric + [str(i+1)  for i in range(100)]
        
        texts = {}
        pattern = '_{4,}'
        for fname in os.listdir(data_folder):
            f = open(data_folder + fname, 'r')
            for l in f.readlines():
                l = re.sub(pattern, ' ', l)
                vid, clip, _, _, text = l.strip().split('___')
                texts[str(vid) + '_' + str(clip)] = preprocess(text, valids)

        for path,vid,clip,emotion,arousal,valence in metadata.values:
            if(str(vid) + '_' + str(clip) in texts.keys()):
                words = texts[str(vid) + '_' + str(clip)]
                n = len(words)
                if(len(words) > 0):
                    data = self.df_sentic.loc[words, :]
                    features_i = np.percentile(data[numeric], (0,25,50,75,100), 0).ravel()
                    # features_i = data[numeric].sum().values.ravel()
                    # features_i = data[numeric].mean().values.ravel()
                    tag_to_id = {'#joy':0, '#sadness':1, '#interest':2, '#surprise':3, '#anger':4, '#fear':5, '#admiration':6, '#disgust':7}
                    tags = np.zeros((8), dtype=np.float32)
                    for k,v in data['primary_mood'].value_counts().items():
                        tags[tag_to_id[k]] = v/n
                    features_i = np.append(features_i, tags, axis=0)

                    tags = np.zeros((8), dtype=np.float32)
                    for k,v in data['secondary_mood'].value_counts().items():
                        tags[tag_to_id[k]] = v/n
                    features_i = np.append(features_i, tags, axis=0)
                else:
                    features_i = np.array([None for _ in range(self.n_features)])
            else:
                features_i = np.array([None for _ in range(self.n_features)])
            features.append(features_i)
            labels.append([emotion, arousal, valence])
        
        features = np.array(features)
        labels = np.array(labels)
        return features, labels

    def run_fb(self, metadata):
        features= []
        labels = []
        valids = list(self.df_sentic.index)
        stopwords = [line.rstrip('\n') for line in open('data/stop_words.txt')]
        numeric = ['pleasantness_value', 'attention_value', 'sensitivity_value', 'aptitude_value', 'polarity_value']
        numeric = numeric + [str(i+1)  for i in range(100)]
        cont = 0
        for text,arousal,valence in metadata.values:
            emotion = aro_val_to_emo(arousal, valence)
            words = preprocess(text, valids)
            n = len(words)
            if(len(words) > 0):
                data = self.df_sentic.loc[words, :]
                tag_to_id = {'#joy':0, '#sadness':1, '#interest':2, '#surprise':3, '#anger':4, '#fear':5, '#admiration':6, '#disgust':7}
                mapa = {0:[0,0,0,0,0,0,0], 1:[0,0,0,0,0,0,1], 2:[0,0,0,0,0,1,0], 3:[0,0,0,0,1,0,0], 4:[0,0,0,1,0,0,0], 5:[0,0,1,0,0,0,0], 6:[0,1,0,0,0,0,0], 7:[1,0,0,0,0,0,0]}
                serie = data[numeric].values
                pm = np.array([mapa[tag_to_id[x]] for x in data['primary_mood'].values])
                sm = np.array([mapa[tag_to_id[x]] for x in data['secondary_mood'].values])
                serie = np.append(serie, pm, axis=1)
                serie = np.append(serie, sm, axis=1)
                np.save('data/features/fb_'+str(cont)+'.npy', serie)
                features_i = np.percentile(data[numeric], (0,25,50,75,100), 0).ravel()
                # features_i = data[numeric].sum().values.ravel()
                # features_i = data[numeric].mean().values.ravel()
                tags = np.zeros((8), dtype=np.float32)
                for k,v in data['primary_mood'].value_counts().items():
                    tags[tag_to_id[k]] = v/n
                features_i = np.append(features_i, tags, axis=0)

                tags = np.zeros((8), dtype=np.float32)
                for k,v in data['secondary_mood'].value_counts().items():
                    tags[tag_to_id[k]] = v/n
                features_i = np.append(features_i, tags, axis=0)
            else:
                serie = np.empty((2,119))
                serie[:] = np.nan
                np.save('data/features/fb_'+str(cont)+'.npy', serie)
                features_i = np.array([None for _ in range(self.n_features)])
            features.append(features_i)
            labels.append([emotion, arousal, valence])
            cont += 1
        
        features = np.array(features)
        labels = np.array(labels)
        return features, labels



    
    def select_features_categorical(self, X, y, dataset):
        df_f = pd.DataFrame(X)
        df_f.columns = np.arange(df_f.shape[1])
        print("features: ", len(df_f.columns))
        epsilon = 1e-5
        df_f_tmp = df_f.loc[:,df_f.var()>epsilon]
        tmp = list(df_f_tmp.columns)
        df_f_tmp.columns = np.arange(df_f_tmp.shape[1])

        X = df_f_tmp.values.astype(np.float32)

        # remove nans
        idxs = []
        for i in range(len(X)):
            if(not np.isnan(X[i,:]).any()):
                idxs.append(i)
        
        X = X[idxs]
        y = y[idxs](y, dataset)

        X, tmp = self.select_vif(X, tmp)
        df_f = df_f.loc[:, tmp]
        print("selected after VIF filter:", len(tmp))

        clf = RandomForestClassifier()
        clf.fit(X,y)
        
        self.idxs = []
        threshold = 1e-3
        importances = clf.feature_importances_
        for i in range(len(importances)):
            if(importances[i]>threshold):
                self.idxs.append(tmp[i])
        self.idxs = np.array(self.idxs)
        df_f = df_f.loc[:, importances>threshold]
        print("selected:", len(df_f.columns))

        # df = pd.concat((df_m, df_f), axis=1, ignore_index=True)
        # df.columns = np.arange(df.shape[1])
        X = df_f.values

        return X
    
    def select_features_continuous(self, X, y, dataset):
        X = X.astype(np.float32)
        df_f = pd.DataFrame(X)
        df_f.columns = np.arange(df_f.shape[1])

        print("features: ", len(df_f.columns))

        epsilon = 1e-4 # 1e-5
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

        return [df_f.values, df_f.values, df_f.values]

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
                elif(len(tmp) > 150):
                    limit = 5
                
                data = [[i, vif] for i,vif in enumerate(vifs)]
                data = sorted(data, key=lambda x: x[1], reverse=True)
                idxs = [x[0] for x in data[:limit]]

                X = np.delete(X, idxs, axis=1)
                tmp = np.delete(tmp, idxs)

            else:
                if(np.max(vifs) > 10):
                    idx = np.argmax(vifs)
                    tmp = np.delete(tmp, idx)
                    X = np.concatenate((X[:, :idx], X[:, idx+1:]), axis=1)
        
        return X, tmp


