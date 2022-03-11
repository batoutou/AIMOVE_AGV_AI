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
from manager.actions import *

# Mode = "TRAIN_MODE"
Mode = "TEST_MODE"
Model = "FCNN"  # GMM
FPS_selected = 10
ratio_image = 1.3

time_frame = collections.deque(maxlen=30)

previousTime = 0

buff_np=np.zeros([30,42])

FCNN = keras.models.load_model('FCNN/FCNN_model')

classes = class_extract('data/train')

cap = cv2.VideoCapture(0)   # For webcam input, 0 original webcam, q1 extern webcam

# To order the actions
act = action(classes)

while True:#cap.isOpened():

    success, image = cap.read()

    if not success:
        # print("Ignoring empty camera frame.")
        continue

    hands = mp_model()
    
    image.flags.writeable = True
    
    # results = hands.process(image)q
    image, results = mediapipe_detection(image, hands)

    image = draw_landmark(image, results)

    list_joints_image = exctract_joint_image(image, results)

    C=" "
    
    if(len(list_joints_image)==42):
        list_joints_image=pre_process_landmark(list_joints_image)

        probs = predict_model_FCNN(FCNN, list_joints_image) 
        
        C=classes[int(probs)]
        
    C = mean_classes(C,classes)

    message = act.add_action(C)

    time_prog, previousTime, image = FPS(message, C, previousTime, image)

    image = cv2.resize(image, (0,0), fx=ratio_image, fy=ratio_image) 

    cv2.imshow('MediaPipe Hands', image)

    sleep(time_prog, FPS_selected)

    pressedKey = cv2.waitKey(1) & 0xFF

    if pressedKey == ord("q"): 
        cap.release()
        break
