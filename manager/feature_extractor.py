import mediapipe as mp
import numpy as np
import pandas as pd
import cv2
import tqdm
import glob
import copy


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def mp_model():
    hands= mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
    return hands

def extract_joint_dataset(train_dir):
    train_arrays_lengths=[]
    list_gesture=[]
    for gesture in tqdm(range(len(train_dir))):
        for filename in glob.iglob(str(train_dir[gesture])[2:-2]+'\\\\*', recursive=True):
            V=read_video(filename).to_numpy()
            continue
        V=V[:, 1:]
        print(V.shape)
        #V = np.expand_dims(V, axis = 0)
        list_gesture.append(V)
        train_arrays_lengths.append([V.shape[0]])
        
    train_arrays= np.array(list_gesture, dtype=object)
    return train_arrays

def exctract_joint_image(image, results):

    #results = hands.process(image)

    if results.multi_hand_landmarks:
      image_height, image_width, _ = image.shape

    L1=[]
    if results.multi_hand_landmarks:
      for i in range(21):
          try :
          #results.multi_handedness[0].classification[0].label
              L1.append(results.multi_hand_landmarks[0].landmark[i].x * image_width)
              L1.append(results.multi_hand_landmarks[0].landmark[i].y * image_height)
          except:
              L1.append(np.nan)
              L1.append(np.nan)
    return L1

def draw_landmark(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    return image

def normlization(L1):
    L=copy.copy(L1)
    x_base,y_base=0,0
    for i in range(0,len(L),2):
        if(i==0):
            x_base,y_base=L[0],L[1]
            L[0],L[1]=0,0
        else:
            L[i]=L[i]-x_base
            L[i+1]=L[i+1]-y_base
    return L

def read_video(path,hands):
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(path)

    #construct head of the dataframe
    title=['frame']
    for i in range(21):
        title.append(str(i)+'_x')
        title.append(str(i)+'_y')
    df = pd.DataFrame(columns=title)

    idx=0

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                image_height, image_width, _ = image.shape

            if results.multi_hand_landmarks:
                L1=[]
                for i in range(21):
                    try :
                    #results.multi_handedness[0].classification[0].label
                        L1.append(results.multi_hand_landmarks[0].landmark[i].x * image_width)
                        L1.append(results.multi_hand_landmarks[0].landmark[i].y * image_height)
                    except:
                        L1.append(np.nan)
                        L1.append(np.nan)
                L1=normlization(L1)
                df_temp = pd.Series(L1, index = df.columns)
                df.loc[len(df)] = df_temp
                idx+=1
            
            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord("q"):  # Record pressing r
                break

    cap.release()
    return df


# path_image = r"C:\Users\franc\Desktop\RT_Detection_AGV\data\WIN_20220223_15_50_06_Pro.JPG"

# im = cv2.imread(path_image)

# print(exctract_joint_image(im,mp_model()))