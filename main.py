from attr import define

import sys 
import os

import cv2
import collections
import numpy as np
import pandas as pd

from GMM.model_GMM import *
from FCNN.model_FCNN import *
from Mediapipe.webcam_manager import *

from data import *

Mode = "TRAIN_MODE"  # "TEST_MODE"
Model = "FCNN"  # GMM

path_train = "./data/train"# "C:\Users\bapti\OneDrive - mines-paristech.fr\Year project\Code\AIMOVE_AGV_AI\data\train"
path_test = './data/test'

time_frame = collections.deque(maxlen=30)

previousTime = 0

buff_np=np.zeros([30,42])

if (Mode == "TRAIN_MODE"):
    # TODO
    # train_arrays = read_pickle(file_name = "save_train_data.pkl")
    with open(path_train, 'r') as f:
        train_arrays=np.load(f)
    
    print(train_arrays.shape)
    
    sys.exit()

    if(train_arrays.shape[0] == 0): #Si y a pas de données dans le pickle
        print("No data in the pinkle file")
        train_arrays = data_extraction(path_test)
        # TODO
        save_pickle(file_name = "save_train_data.pkl")        

    if(Model == "GMM"):
        train_arrays = arrange_data_for_GMM(train_arrays)
        GMM = train_model_GMM(train_arrays)
        print("GMM trained")
        
    if(Model == "FCNN"):
        train_arrays = arrange_data_for_FCNN(train_arrays)
        FCNN = train_model_FCNN(train_arrays)
        print("FCNN trained")
        



cap = cv2.VideoCapture(0)   # For webcam input

while True:#cap.isOpened():

    success, image = cap.read()

    if not success:
        print("Ignoring empty camera frame.")
        continue

    hands = mp_model()
    
    image.flags.writeable = True
    
    results = hands.process(image)

    image = draw_landmark(image, results)

    list_joints_image = exctract_joint_image(image, results)

    list_joints_image = normlization(list_joints_image)

    buff_np = np.array(time_frame.append(list_joints_image))

    predict_model_GMM(GMM, np.array(list_joints_image))
    
    # idx+=1
    
    # if (idx>30):
    #     resultats=[]
    #     for hmm_model_num in range(len(hmm_models)):
    #         res=hmm_models[hmm_model_num].score(np.array(buff_np))
    #         resultats.append(res)
    # print(resultats)
    # ide=resultats.index(max(resultats))
    # print("Best model : ", classes[ide])


    # seuil=-20000
    # if (max(resultats) > seuil):
    #   ide=resultats.index(max(resultats))
    #   print("Best model : ", classes[ide])
    # else:
    #   print("No matching signs ")

    previousTime, image = FPS(previousTime, image)

    cv2.imshow('MediaPipe Hands', image)

    pressedKey = cv2.waitKey(1) & 0xFF

    if pressedKey == ord("q"): 
        cap.release()
        break
