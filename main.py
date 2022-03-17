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
from class_camera import *

# Mode = "TRAIN_MODE"
Mode = "TEST_MODE"
FPS_selected = 10
ratio_image = 1.3
path_train = "./data/train"
path_test = "./data/test"
path_train_pkl= "./data/save_train_data.pkl"
previousTime = 0
list_cam=[]

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


context = zmq.Context()

print("Connecting to hello world server…")  #  Socket to talk to server
socket = context.socket(zmq.REQ)
# socket.connect("tcp://172.20.10.2:5555")
socket.connect("tcp://172.20.10.14:5555")

video_stream_widget = VideoStreamWidget(0)
video_stream_widget1 = VideoStreamWidget(1)

act = action(classes)  # To order the actions

while True: #cap.isOpened():
    classe_detected_1 = video_stream_widget.detect_class(FCNN,classes)
    classe_detected_2 = video_stream_widget1.detect_class(FCNN,classes)
    
    video_stream_widget.show_frame()
    video_stream_widget1.show_frame()

    message1_cv2, message1_robot = act.add_action(classe_detected_1)
    message2_cv2, message2_robot = act.add_action(classe_detected_2)
    # print(message1_cv2, message1_robot, message2_cv2, message2_robot)

    if(message1_robot!=-1): send_message(socket, message1_robot)
    if(message2_robot!=-1): send_message(socket, message2_robot)

    # time_prog, previousTime, image = FPS(message_cv2, C, previousTime, image)

    # image = cv2.resize(image, (0,0), fx=ratio_image, fy=ratio_image) 

    # cv2.imshow('MediaPipe Hands', image)

    # sleep(time_prog, FPS_selected)

    # pressedKey = cv2.waitKey(1) & 0xFF

    # if pressedKey == ord("q"): 
    #     cap.release()
    #     break
