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
FPS_selected = 10
ratio_image = 1.3
path_train = "./data/train"
path_test = "./data/test"
path_train_pkl= "./data/save_train_data.pkl"
previousTime = 0


if (Mode == "TRAIN_MODE"):
    if os.path.isfile(path_train_pkl) == True:  #Si y a des données dans le pickle
        X_train, Y_train = read_pickle(path_train_pkl)
    else:
        print("No data in the pinkle file")
        X_train, Y_train = data_extraction(path_train)
        save_pickle([X_train, Y_train], path_train_pkl)
    
    FCNN = train_model_FCNN(X_train, Y_train)
    print("FCNN trained")
    FCNN.save('FCNN/FCNN_model')
    print("FCNN saved")
        
if(Mode == "TEST_MODE"): FCNN = keras.models.load_model('./FCNN/FCNN_model')

classes = class_extract('./data/train')
print("classes : ",classes)
# sys.exit()

cap = cv2.VideoCapture(0)   # For webcam input, 0 original webcam, q1 extern webcam

act = action(classes)  # To order the actions

import zmq
context = zmq.Context()
#  Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://192.168.137.121:5555")

while True:#cap.isOpened():

    success, image = cap.read()

    if not success:
        print("Ignoring empty camera frame.")
        continue

    hands = mp_model()
    
    image.flags.writeable = True

    image, results = mediapipe_detection(image, hands)

    image = draw_landmark(image, results)

    list_joints_image = exctract_joint_image(image, results)

    C="No class detected"

    if(len(list_joints_image)==42):
        list_joints_image=pre_process_landmark(list_joints_image)

        probs = predict_model_FCNN(FCNN, list_joints_image) 
        
        C=classes[int(probs)]

    C = mean_classes(C,classes)

    message_cv2, message_robot = act.add_action(C)

    if(message_robot!=-1):
        print("Sending request {}".format(message_robot))
        socket.send(message_robot.encode())
        
        message = socket.recv()#  Get the reply.
        print("Received reply %s" % (message))

    time_prog, previousTime, image = FPS(message_cv2, C, previousTime, image)

    image = cv2.resize(image, (0,0), fx=ratio_image, fy=ratio_image) 

    cv2.imshow('MediaPipe Hands', image)

    sleep(time_prog, FPS_selected)

    pressedKey = cv2.waitKey(1) & 0xFF

    if pressedKey == ord("q"): 
        cap.release()
        break
