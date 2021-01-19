import numpy as np
import os
import pandas as pd
import subprocess
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from utils import emotion_to_id, aro_val_to_emo
from sklearn.model_selection import train_test_split

def parallel(X, i):
    y = X[:, i]
    X_i = np.concatenate((X[:, :i], X[:, i+1:]), axis=1)
    clf = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X_i, y, test_size = 0.3)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

class Featurizer():
    def __init__(self):
        self.n_features = 6374

    def run_enterface(self, metadata):
        # open smile, compare feature set
        df = pd.DataFrame()
        labels = []
        banned = ['subject 6', 'subject 23']
        for subject, emotion, sentence, path in metadata.values:
            if(not subject in banned):
                path = '../' + path.replace('.avi', '.wav')
                data_fname = 'data/csvs/' + subject[subject.rindex(' ')+1:] + '_' + emotion + '_' + sentence[-1] + '.csv'

                if(not os.path.isfile(data_fname)):
                    subprocess.run(['SMILExtract', '-C', 'data/config/IS13_ComParE.conf', '-I', path, '-csvoutput', data_fname])
                
                df_tmp = pd.read_csv(data_fname, sep=';', header=0)
                df_tmp = df_tmp.drop(list(df_tmp.columns)[0], axis=1)
                df = df.append(df_tmp, ignore_index=True)
                labels.append(emotion)
        return df.values, labels
    
    def run_iemocap(self, metadata):
        data_folder = '../data/IEMOCAP_full_release/'
        df = pd.DataFrame()
        labels = []

        for sess,fname,utt,_,_,arousal,valence,emotion in metadata.values:
            data_fname = 'data/csvs/' + fname + '_' + utt + '.csv'

            if(not os.path.isfile(data_fname)):
                path = data_folder + sess + '/sentences/wav/' + fname +'/' + fname + '_' + utt + '.wav'
                subprocess.run(['SMILExtract', '-C', 'data/config/IS13_ComParE.conf', '-I', path, '-csvoutput', data_fname])
            
            df_tmp = pd.read_csv(data_fname, sep=';', header=0)
            df_tmp = df_tmp.drop(list(df_tmp.columns)[0], axis=1)
            df = df.append(df_tmp, ignore_index=True)
            labels.append([emotion, arousal, valence])
        
        return df.values, np.array(labels)
    
    def run_mosei(self, metadata):
        data_folder = '../data/mosei/Raw/'
        df = pd.DataFrame()
        labels = []

        for path,vid,clip,emotion,arousal,valence in metadata.values:
            path = '../' + path.replace('[modal]', 'Audio').replace('[ext]', '.wav')
            if(os.path.isfile(path)):
                data_fname = 'data/csvs/mosei' + str(vid) + '_' + str(clip) + '.csv'

                if(not os.path.isfile(data_fname)):
                    subprocess.run(['SMILExtract', '-C', 'data/config/IS13_ComParE.conf', '-I', path, '-csvoutput', data_fname])

                df_tmp = pd.read_csv(data_fname, sep=';', header=0)
                df_tmp = df_tmp.drop(list(df_tmp.columns)[0], axis=1)
            else:
                df_tmp = pd.DataFrame([np.nan for _ in range(self.n_features)])
            df = df.append(df_tmp, ignore_index=True)
            labels.append([emotion, arousal, valence])
        
        return df.values, np.array(labels)
    
    def run_hci(self, metadata):
        data_folder = '../data/hci_tagging/Sessions/'
        df = pd.DataFrame()
        labels = []

        for sess, aro, val in metadata.values:
            emotion = aro_val_to_emo(aro, val)
            wavs = [x for x in os.listdir(data_folder+'/'+str(sess)) if '.wav' in x]
            if(len(wavs) == 1):
                data_fname = data_folder+'/'+str(sess)+'/'+wavs[0]
                csv_fname = 'data/csvs/hci_'+str(sess)+'.csv'
                if(not os.path.isfile(csv_fname)):
                    subprocess.run(['SMILExtract', '-C', 'data/config/IS13_ComParE.conf', '-I', data_fname, '-csvoutput', csv_fname])
                
                df_tmp = pd.read_csv(csv_fname, sep=';', header=0)
                df_tmp = df_tmp.drop(list(df_tmp.columns)[0], axis=1)
            elif(len(wavs)>1):
                raise Exception('more than 1 wav file for session: ' + str(sess))
            else:
                df_tmp = pd.DataFrame([np.nan for _ in range(self.n_features)])
            df = df.append(df_tmp, ignore_index=True)
            labels.append([emotion, aro, val])
        
        return df.values, np.array(labels)
    
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
        df_f = pd.DataFrame(X)
        df_f.columns = np.arange(df_f.shape[1])

        print("features: ", len(df_f.columns))

        epsilon = 1e-4 # 1e-5
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
        y = np.array(y)[idxs]
        y = emotion_to_id(y, dataset)

        X, tmp = self.select_vif(X, tmp)
        df_f = df_f.loc[:, tmp]
        print("selected after VIF filter:", len(tmp))

        clf = RandomForestClassifier()
        clf.fit(X,y)

        self.idxs = []
        # threshold = 1e-3
        threshold = 5e-4
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
                    limit = 1000
                elif(len(tmp) > 500):
                    limit = 300
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
        