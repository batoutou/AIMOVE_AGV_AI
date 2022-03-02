import glob
import os
import numpy as np
import copy as copy
from tqdm import tqdm
import pickle
from sklearn import mixture

from feature_extractor import *

nb_classe=2

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

#hmm_models

def arrange_data_for_FCNN(train_arrays):
    data = np.empty((0,42))
    for i in range(train_arrays.shape[0]): data=np.append(data,train_arrays[i], axis=0)
    return data#sum([train_arrays[i].shape[0] for i in range(train_arrays.shape[0])])

def train_model_FCNN(data_train):
    fcnn = mixture.GaussianMixture(n_components=nb_classe, max_iter=1000, covariance_type='full').fit(data_train)
    return fcnn

def predict_model_FCNN(fcnn, data_test):
    probs = fcnn.predict_proba(data_test)
    return probs

def read_video(path):
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(path)
    data = np.empty((0,42))
    hands = mp_model()
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            #print("Ignoring empty camera frame.")
            break
        results = hands.process(image)
        L1=exctract_joint_image(image, results)
        if(len(L1)==42):
            L1=normlization(L1)
            data = np.append(data, np.array([L1]), axis=0)
    cap.release()
    return data

# path = r'C:\Users\franc\Desktop\RT_Detection_AGV\data\test\five\test1.mp4'
# a=read_video(path)
# print(a.shape)

    

