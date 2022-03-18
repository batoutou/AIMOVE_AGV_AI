
from threading import Thread
import cv2
import collections
from datetime import datetime
import argparse
import cv2

import cv2
from FCNN.model_FCNN import *
from manager.webcam_manager import *
from manager.i_o_manager import *
from manager.actions import *
import keras

buff = collections.deque(maxlen=10)

from threading import Thread
import cv2, time

def putIterationsPerSec(frame, iterations_per_sec):
    cv2.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
        (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame

def putClasse(frame, max_buff):
    cv2.putText(frame, str(max_buff),
        (10, 425), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame

class CountsPerSec:
    def __init__(self):
        self._start_time = None
        self._num_occurrences = 0

    def start(self):
        self._start_time = datetime.now()
        return self

    def increment(self):
        self._num_occurrences += 1

    def countsPerSec(self):
        elapsed_time = (datetime.now() - self._start_time).total_seconds()
        return self._num_occurrences / elapsed_time

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

        self.buff = collections.deque(maxlen=10)
        self.results=0
        self.max_buff=0

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True

    def mean_classes(self, C):
        self.buff.append(C)
        list_sorted =  Counter(self.buff)
        return max(list_sorted.items(), key=operator.itemgetter(1))[0]

    def detect_class(self, FCNN, classes):
        hands = mp_model()
        self.frame.flags.writeable = True
        self.frame, results = mediapipe_detection(self.frame, hands)
        # self.frame = draw_landmark(self.frame, results)
        list_joints_image = exctract_joint_image(self.frame, results)
        C="No class detected"
        if(len(list_joints_image)==42):
            list_joints_image=pre_process_landmark(list_joints_image)
            probs = predict_model_FCNN(FCNN, list_joints_image)
            C=classes[int(probs)]
        self.max_buff = self.mean_classes(C)
        return self.max_buff


if __name__ == "__main__":

    list_camera=[0,1]
    video_getter_list=[]

    FCNN = keras.models.load_model(r'FCNN\FCNN_model')
    classes = class_extract(r'data\train')
    print("classes : ", classes)

    for i in range(len(list_camera)): video_getter_list.append(VideoGet(list_camera[i]).start())

    cps = CountsPerSec().start()

    while True:

        if (cv2.waitKey(1) == ord("q")):
            for i in range(len(list_camera)): video_getter_list[i].stop()
        
        frame_list=[]
        classe_detected_list=[]
        for i in range(len(list_camera)): 
            frame_list.append(video_getter_list[i].frame)
            classe_detected_list.append(video_getter_list[i].detect_class(FCNN, classes))
            frame_list[i] = putIterationsPerSec(frame_list[i], cps.countsPerSec())
            frame_list[i] = putClasse(frame_list[i], classe_detected_list[i])
            cv2.imshow("Video{}".format(i), frame_list[i])
        cps.increment()