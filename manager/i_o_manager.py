
import glob
import os
import numpy as np
import copy as copy
from tqdm import tqdm
import pickle
from sklearn import mixture
import copy, itertools

from manager.feature_extractor import *

def save_pickle(train_arrays, file_name):
    with open(file_name, "wb") as fp:
        pickle.dump(train_arrays, fp)

def read_pickle(file_name): #To read pinkle file
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

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def data_extraction(path):
    train_arrays_X=np.empty((0,42))
    list_gesture_Y=[]
    train_dir=path_extraction(path)
    for gesture in tqdm.tqdm(range(len(train_dir))):
        for filename in glob.iglob(str(train_dir[gesture])[2:-2]+'\\\\*', recursive=True):
            X=read_video(filename)
            print("Video extraite de : ",filename)
            continue
        #V = np.expand_dims(V, axis = 0)
        Y=[]
        for i in range(X.shape[0]): 
            Y.append(gesture)

        train_arrays_X=np.concatenate((train_arrays_X, X), axis=0)
        list_gesture_Y=list_gesture_Y + Y
    return train_arrays_X, list_gesture_Y

def read_video(path):
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(path)
    data = np.empty((0,42))
    hands = mp_model()
    
    # tqdm.tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            #print("Ignoring empty camera frame.")
            break
        # results = hands.process(image)
        image, results = mediapipe_detection(image, hands)
        L1=exctract_joint_image(image, results)
        if(len(L1)==42):
            # L1=normlization(L1)
            L1=pre_process_landmark(L1)
            data = np.append(data, np.array([L1]), axis=0)
    cap.release()
    return data

def send_message(socket, message_robot):
    print("Sending request {}".format(message_robot))
    socket.send(message_robot.encode())
    
    message = socket.recv()#  Get the reply.
    print("Received reply %s" % (message))
    # return message

def pre_process_landmark(L):
    landmark_list=np.array([L]).reshape((21, 2))
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list