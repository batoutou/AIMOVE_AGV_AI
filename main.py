from attr import define

import sys 
import os
import zmq
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
# from class_camera import *
from test import *

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
        
if(Mode == "TEST_MODE"): FCNN = keras.models.load_model(r'FCNN\FCNN_model')

classes = class_extract(r'data\train')
print("classes : ",classes)
# sys.exit()

# Gère la connection TCP/IP
context = zmq.Context()
print("Connecting to hello world server…")  #  Socket to talk to server
socket = context.socket(zmq.REQ)
socket.connect("tcp://172.20.10.2:5555") #gautier
# socket.connect("tcp://172.20.10.14:5555") #françois 

#Gère une liste de webcam
list_camera=[0,1]
action_administrator_liste=[]
video_getter_list=[]

for i in range(len(list_camera)): 
    video_getter_list.append(VideoGet(list_camera[i]).start())
    action_administrator_liste.append(action(classes))#To order the actions

cps = CountsPerSec().start()

while True:
    if (cv2.waitKey(1) == ord("q")):
        for i in range(len(list_camera)): video_getter_list[i].stop()
    
    frame_list=[]
    classe_detected_list=[]
    message_cv2_list=[]
    message_robot_list=[]

    for i in range(len(list_camera)): 
        frame_list.append(video_getter_list[i].frame)
        classe_detected_list.append(video_getter_list[i].detect_class(FCNN, classes))

        frame_list[i] = putIterationsPerSec(frame_list[i], cps.countsPerSec())
        frame_list[i] = putClasse(frame_list[i], classe_detected_list[i])
        
        _, X = action_administrator_liste[i].add_action(classe_detected_list[i])
        message_robot_list.append(X)

        if(message_robot_list[i]!=-1): send_message(socket, message_robot_list[i])

        cv2.imshow("Video{}".format(i), frame_list[i])

    cps.increment()