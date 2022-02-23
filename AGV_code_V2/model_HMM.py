import glob
import os
import numpy as np
import copy as copy
from tqdm import tqdm
import pickle
from sklearn import mixture

nb_classe=2

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

path = r'C:\Users\franc\Desktop\RT_Detection_AGV\data\train'

#hmm_models

def save_pickle(file_name = "save_train_data.pkl"): #To save as pinkle
    file_name = "save_train_data.pkl"
    train_arrays=np.array([])
    open_file = open(file_name, "wb")
    pickle.dump(train_arrays, open_file)
    open_file.close()

def read_pickle(file_name = "save_train_data.pkl"): #To read pinkle file
    open_file = open(file_name, "rb")
    train_arrays = pickle.load(open_file)
    open_file.close()
    return train_arrays

def class_extract(path):
    classes = next(os.walk( path) )
    classes=classes[1]
    return classes

def path_extraction(path):
    classes=class_extract(path)
    train_dir=[]
    for num_classes in range(len(classes)):
        train_dir.append(glob.glob(path+"\\"+classes[num_classes]))
    return train_dir

# def model_HMM():
#     resultats=[]
#     for hmm_model_num in range(len(hmm_models)):
#         res=hmm_models[hmm_model_num].score(np.array(buff_np))
#         resultats.append(res)
#     return resultats

def train_model_GMM(data_train):
    gmm = mixture.GaussianMixture(n_components=nb_classe, max_iter=1000, covariance_type='full').fit(data_train)
    return gmm

def predict_model_GMM(gmm, data_test):
    probs = gmm.predict_proba(data_test)
    return probs

    

