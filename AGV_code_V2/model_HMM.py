import glob
import os
import numpy as np
import copy as copy
from tqdm import tqdm
import pickle
from sklearn import mixture

from mediapipe1 import *

nb_classe=2

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

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

def data_extraction(path):
    gestures_list=[]
    list_gesture=[]
    train_dir=path_extraction(path)
    for gesture in range(len(train_dir)):
        for filename in glob.iglob(str(train_dir[gesture])[2:-2]+'\\\\*', recursive=True):
            V=read_video(filename)
            print("Video extraite de : ",filename)
            continue
        #V = np.expand_dims(V, axis = 0)
        list_gesture.append(V)
    train_arrays= np.array(list_gesture, dtype=object)
    return train_arrays


# def model_HMM():
#     resultats=[]
#     for hmm_model_num in range(len(hmm_models)):
#         res=hmm_models[hmm_model_num].score(np.array(buff_np))
#         resultats.append(res)
#     return resultats

def arrange_data_for_GMM(train_arrays):
    print(train_arrays.shape)
    print(train_arrays[0].shape[0]+train_arrays[1].shape[0])
    train_arrays=train_arrays.reshape(train_arrays[0].shape[0]+train_arrays[1].shape[0],42)
    print(train_arrays.shape)
    return train_arrays

def train_model_GMM(data_train):
    gmm = mixture.GaussianMixture(n_components=nb_classe, max_iter=1000, covariance_type='full').fit(data_train)
    return gmm

def predict_model_GMM(gmm, data_test):
    probs = gmm.predict_proba(data_test)
    return probs

def read_video(path):
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(path)

    data = np.empty((0,42))
    print(data.shape)

    hands = mp_model()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        results = hands.process(image)
        L1=exctract_joint_image(image, results)
        if(len(L1)==42):

            L1=normlization(L1)
            print(len(L1))
            data = np.append(data, np.array([L1]), axis=0)
        #print(data.shape)

    cap.release()
    print("fin de read video")
    return data

# path = r'C:\Users\franc\Desktop\RT_Detection_AGV\data\test\five\test1.mp4'
# a=read_video(path)
# print(a.shape)

    

