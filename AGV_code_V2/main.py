from attr import define
import cv2
import collections
import numpy as np
import pandas as pd

from model_HMM import *
from mediapipe1 import *
from webcam_manager import *

Mode = "TEST_MODE"
Model = "GMM"

d = collections.deque(maxlen=30)

cap = cv2.VideoCapture(0)   # For webcam input

previousTime = 0

buff_np=np.zeros([30,42])

if (Mode == "TRAIN_MODE"):
    train_arrays=read_pickle(file_name = "save_train_data.pkl")
    if(Mode == "GMM"):
        GMM = train_model_GMM()



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

    buff_np = np.array(d.append(list_joints_image))

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
