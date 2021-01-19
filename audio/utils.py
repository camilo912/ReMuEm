import numpy as np

def standarize(data):
    means = np.mean(data, axis=0).reshape(1,-1)
    stds = np.std(data, axis=0).reshape(1,-1)
    data_s = (data-means)/stds
    return data_s, means, stds

def category_to_id(cats, null=None):
    uniques = np.unique(cats)
    if(null is not None):
        uniques = np.array([x for x in uniques if x != null])
    dic = {x:i for i,x in enumerate(np.unique(cats))}
    if(null is not None):
        ns_id = len(dic)
        dic[null] = ns_id
    func_vec = np.vectorize(lambda x: dic[x])
    return func_vec(cats), ns_id

def emotion_to_id(cats, dataset):
    if(dataset == 'enterface'):
        mapa = {'anger':0, 'disgust':1, 'fear':2, 'happiness':3, 'sadness':4, 'surprise':5}
    elif(dataset == 'iemocap'):
        mapa = {'ang': 0, 'dis':1, 'exc':2, 'fea':3, 'fru':4, 'hap':5, 'neu':6, 'sad':7, 'sur':8, 'oth':9, 'xxx':-1}
    elif(dataset == 'mosei'):
        mapa = {'ang':0, 'dis':1, 'fea':2, 'hap':3, 'sad':4, 'sur':5, 'neu':6, 'mul':-1}
    elif(dataset == 'hci'):
        mapa = {'anger':0, 'disgust':1, 'fear':2, 'happiness':3, 'sadness':4, 'surprise':5, 'other':6}
    func_vec = np.vectorize(lambda x: mapa[x])
    return func_vec(cats)

def fuse_arousal_valence_data(X_aro, X_val):
    for i in range(X_val.shape[1]):
        band = True
        for j in range(X_aro.shape[1]):
            a = X_aro[:,j]
            b = X_val[:,i]
            a = np.nan_to_num(a)
            b = np.nan_to_num(b)
            if(np.array_equal(a,b)):
                band = False
        if(band):
            X_aro = np.concatenate((X_aro, X_val[:,[i]]),axis=1)
    
    return X_aro

def get_angle(x,y):
    if(x == 0):
        if(y == 0):
            return 0
        elif(y > 0):
            return 90
        else:
            return 270
    elif(y == 0):
        if(x > 0):
            return 0
        else:
            return 180
    h = (x**2 + y**2)**(1/2)

    # relative angle
    rel_ang = np.arccos((x**2+h**2-y**2) / (2*x*h))

    # angle to positive x axis
    if(x > 0):
        if(y > 0):
            return rel_ang
        else:
            return 360 - rel_ang
    else:
        if(y > 0):
            return 180 - rel_ang
        else:
            return rel_ang + 180


def aro_val_to_emo(aro, val):
    mapa = {'anger':[-0.51, 0.59], 'disgust':[-0.6, 0.35], 'fear':[-0.64, 0.6], 'happiness':[0.81, 0.51], 'sadness':[-0.63, -0.27], 'surprise':[0.4, 0.67]}
    emo = 'other'
    mini = np.inf
    tol = 15
    for k,v in mapa.items():
        ang_emo = get_angle(*v)
        ang = get_angle(val, aro)
        
        dist = np.linalg.norm(np.array([val,aro])-np.array(v))
        if(dist < mini and ang_emo - tol < ang and ang < ang_emo + tol):
            mini = dist
            emo = k

    return emo
