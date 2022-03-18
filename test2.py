import cv2
from FCNN.model_FCNN import *
from manager.webcam_manager import *
from manager.i_o_manager import *
from manager.actions import *
import keras

buff = collections.deque(maxlen=10)

from threading import Thread
import cv2, time

class fps(object):
    def __init__(self):
        self.FPS = None
        self.previousTime = 0
        self.currentTime = None
        self.t_programme = None
    
    def calcul_fps(self):
        self.currentTime = time.time()
        self.t_programme=self.currentTime-self.previousTime
        self.FPS = round(1 / (self.currentTime-self.previousTime))
        self.previousTime = self.currentTime
        # return self.FPS, self.t_programme


class VideoStreamWidget(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.thread = Thread(target=self.update, args=()) # Start the thread to read frames from the video stream
        self.thread.daemon = True
        self.thread.start()

        self.buff = collections.deque(maxlen=10)
        self.results=0
        self.image_height, self.image_width,_ = self.height_width()
        self.max_buff=0

        # self.fps_obj = fps()

    def update(self):  # Read the next frame from the stream in a different thread 
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                # start_time = time.time()
                # while((time.time() - start_time) < ((1/self.FPS_selected)*1000)):
                #     continue

    def show_frame(self):
        cv2.putText(self.frame, str(self.max_buff), (10,  self.image_height-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Windows webcam nb {}'.format(self.src), self.frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

    def height_width(self):
        _, image = self.capture.read()
        return image.shape

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

# if __name__ == '__main__':
#     F=fps()
#     while True:
#         try:
#                 print(F.calcul_fps())
#                 time.sleep(0.1)
#         except AttributeError:
#             pass

if __name__ == '__main__':
    video_stream_widget = VideoStreamWidget(0)
    video_stream_widget1 = VideoStreamWidget(1)

    FCNN = keras.models.load_model(r'FCNN\FCNN_model')
    classes = class_extract(r'data\train')
    print("classes : ", classes)

    while True:
        try:
            video_stream_widget.detect_class(FCNN,classes)
            video_stream_widget1.detect_class(FCNN,classes)
            
            video_stream_widget.show_frame()
            video_stream_widget1.show_frame()
        except AttributeError:
            pass