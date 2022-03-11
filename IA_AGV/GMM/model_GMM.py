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



# def model_HMM():
#     resultats=[]
#     for hmm_model_num in range(len(hmm_models)):
#         res=hmm_models[hmm_model_num].score(np.array(buff_np))
#         resultats.append(res)
#     return resultats

def arrange_data_for_GMM(train_arrays):
    data = np.empty((0,42))
    for i in range(train_arrays.shape[0]): data=np.append(data,train_arrays[i], axis=0)
    return data#sum([train_arrays[i].shape[0] for i in range(train_arrays.shape[0])])

def train_model_GMM(data_train):
    gmm = mixture.GaussianMixture(n_components=nb_classe, max_iter=1000, covariance_type='full').fit(data_train)
    return gmm

def predict_model_GMM(gmm, data_test):
    probs = gmm.predict_proba(data_test)
    return probs

# path = r'C:\Users\franc\Desktop\RT_Detection_AGV\data\test\five\test1.mp4'
# a=read_video(path)
# print(a.shape)

    

