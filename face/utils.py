import math
import numpy as np

### code adapted from the one created by ApurvaRaj in https://www.geeksforgeeks.org/find-angles-given-triangle/
# returns square of distance b/w two points  
def lengthSquare(X, Y):  
    xDiff = X[0] - Y[0]  
    yDiff = X[1] - Y[1]  
    return xDiff * xDiff + yDiff * yDiff 
      
def get_cosines(A, B, C, h):
    if(A[0] == B[0] and A[1] == B[1]):
        if(B[0]+1 == C[0] and B[1]==C[1]):
            B[0] = B[0] + 2
        else:
            B[0] = B[0] + 2
    if(C[0] == B[0] and C[1] == B[1]):
        if(B[0]+1==A[0] and B[1]==A[1]):
            B[0] = B[0] + 2
        else:
            B[0] = B[0] + 1
    if(A[0] == C[0] and A[1] == C[1]):
        if(C[0]+1 == B[0] and C[1]==B[1]):
            C[0] = C[0] + 2
        else:
            C[0] = C[0] + 1
      
    # Square of lengths be a2, b2, c2  
    a2 = lengthSquare(B, C)  
    b2 = lengthSquare(A, C)  
    c2 = lengthSquare(A, B)  
  
    # length of sides be a, b, c  
    a = math.sqrt(a2);  
    b = math.sqrt(b2);  
    c = math.sqrt(c2);  
  
    # From Cosine law
    alpha = math.acos(min(max((b2 + c2 - a2) / (2 * b * c), -1),1))
    betta = math.acos(min(max((a2 + c2 - b2) / (2 * a * c), -1),1))
    gamma = math.acos(min(max((a2 + b2 - c2) / (2 * a * b), -1),1))
  
    # Converting to degree  
    alpha = alpha * 180 / math.pi
    betta = betta * 180 / math.pi
    gamma = gamma * 180 / math.pi

    # applying cosine transformation
    alpha = np.cos(np.deg2rad(alpha))
    betta = np.cos(np.deg2rad(betta))
    gamma = np.cos(np.deg2rad(gamma))

    return alpha/180, betta/180, gamma/180, a/h, b/h, c/h
### end code adapted from ApurvaRaj

def get_useful_triangles(ls):
    triangles = []
    # triangle coords manually extracted from paper: A fuzzy logic approach to reliable real-time recognition of facial emotion.
    triangles_coords = [[17,19,36], [17,21,37], [19,21,39], [19,36,39], [36,37,40], [38,39,41],
                        [22,24,42], [22,26,44], [24,26,45], [24,42,45], [42,43,46], [43,45,47],
                        [21,22,27], [36,48,60], [39,42,51], [45,54,64], [48,51,54], [48,54,57]]
    for a,b,c in triangles_coords:
        triangles.append([ls[a], ls[b], ls[c]])

    return triangles

def standarize(data):
    means = np.mean(data, axis=0).reshape(1,-1)
    stds = np.std(data, axis=0).reshape(1,-1)
    data_s = (data-means)/stds
    return data_s, means, stds

def extract_closest_face(faces):
    areas = []
    for f in faces:
        areas.append(f.height() * f.width())
    
    idx = np.argmax(areas)
    return faces[idx]

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

def translate(emotion):
    mapa = {'anger':[-0.51, 0.59], 'disgust':[-0.6, 0.35], 'fear':[-0.64, 0.6], 'happiness':[0.81, 0.51], 'sadness':[-0.63, -0.27], 'surprise':[0.4, 0.67]}
    return mapa[emotion]

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
