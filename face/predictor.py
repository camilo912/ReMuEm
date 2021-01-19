import numpy as np
from keras.models import load_model
import time

class Predictor():
    def __init__(self, path, featurizer, idxs, dic):
        self.model = load_model(path)
        featurizer.idxs = idxs
        self.featurizer = featurizer
        self.dic = dic
    
    def predict(self, imgs):
        t0 = time.time()
        features = self.featurizer.featurize_sequence(imgs)
        pred = self.model.predict_classes(np.expand_dims(features, axis=0))
        print(time.time()-t0)
        print(self.dic[int(pred)])
