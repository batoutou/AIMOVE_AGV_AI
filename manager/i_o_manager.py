
import glob
import os
import numpy as np
import copy as copy
from tqdm import tqdm
import pickle
from sklearn import mixture

from manager.feature_extractor import *

def save_pickle(train_arrays, file_name):
    with open(file_name, "wb") as fp:
        pickle.dump(train_arrays, fp)
    #open_file.close()

def read_pickle(file_name): #To read pinkle file
    # open_file = open(file_name, "rb")
    # train_arrays = pickle.load(open_file)
    # open_file.close()

    with open(file_name, "rb") as fp:
        all_data = pickle.load(fp)
    return all_data

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
    Y=[]
    train_dir=path_extraction(path)
    for gesture in range(len(train_dir)):
        Y.append(gesture)
        for filename in glob.iglob(str(train_dir[gesture])[2:-2]+'\\\\*', recursive=True):
            V=read_video(filename)
            print("Video extraite de : ",filename)
            continue
        #V = np.expand_dims(V, axis = 0)
        list_gesture.append(V)
    train_arrays= np.array(list_gesture, dtype=object)
    return train_arrays, Y

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