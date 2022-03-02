from attr import define

import sys 
import os

import cv2
import collections
import numpy as np
import pandas as pd
from tensorflow import keras


# from GMM.model_GMM import *
from FCNN.model_FCNN import *
from manager.webcam_manager import *
from manager.i_o_manager import *

Mode = "TEST_MODE"  # "TEST_MODE"
Model = "FCNN"  # GMM

path_train = "./data/train"
path_test = './data/test'
path_train_pkl= './data/save_train_data.pkl'

time_frame = collections.deque(maxlen=30)

previousTime = 0

buff_np=np.zeros([30,42])

if (Mode == "TRAIN_MODE"):
    if os.path.isfile(path_train_pkl) == True:  #Si y a des données dans le pickle
        X_train, Y_train = read_pickle(path_train_pkl)
    else:
        print("No data in the pinkle file")
        X_train, Y_train = data_extraction(path_train)
        save_pickle([X_train, Y_train], path_train_pkl)

    # if(Model == "GMM"):
    #     train_arrays = arrange_data_for_GMM(train_arrays)
    #     GMM = train_model_GMM(train_arrays)
    #     print("GMM trained")
    
        
    if(Model == "FCNN"):
        FCNN = train_model_FCNN(X_train, Y_train)
        print("FCNN trained")
        FCNN.save('FCNN/FCNN_model')
        print("FCNN saved")
        
if(Mode == "TEST_MODE"):
    FCNN = keras.models.load_model('FCNN/FCNN_model')

print(FCNN.summary())

classes = class_extract('data/test')

# sys.exit()

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

    #buff_np = np.array(time_frame.append(list_joints_image))

    print(np.array(list_joints_image).shape)

    # sys.exit()

    if(len(list_joints_image)==42):
        print(np.array(list_joints_image).shape)
        # PROBLEM (maybe input data shape or type or just the predict function)
        probs = predict_model_FCNN(FCNN, np.array(list_joints_image)) 
        print(probs)

    #predict_model_GMM(GMM, np.array(list_joints_image))
    
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
