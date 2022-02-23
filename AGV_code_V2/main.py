import cv2
import collections
import numpy as np
import pandas as pd

from model_HMM import *
from mediapipe1 import *
from webcam_manager import *

d = collections.deque(maxlen=30)

cap = cv2.VideoCapture(0)   # For webcam input

previousTime = 0

buff_np=np.zeros([30,42])

while True:#cap.isOpened():

    success, image = cap.read()

    if not success:
        print("Ignoring empty camera frame.")
        continue

    hands = mp_model()
    
    image.flags.writeable = True
    
    results = hands.process(image)

    image = draw_landmark(image, results)

    L1 = exctract_joint_image(image, results)

    buff_np = np.array(d.append(normlization(L1)))

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
