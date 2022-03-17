import cv2
from FCNN.model_FCNN import *
from manager.webcam_manager import *
from manager.i_o_manager import *
from manager.actions import *
import keras

buff = collections.deque(maxlen=10)

from threading import Thread
import cv2, time

class VideoStreamWidget(object):
    def __init__(self, src=0):
        self.src=src
        self.capture = cv2.VideoCapture(self.src)
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        self.buff = collections.deque(maxlen=10)
        self.results=0
        self.image_height, self.image_width,_ = self.height_width()
        self.max_buff=0

        self.FPS=0
        self.previousTime=0
        self.currentTime=0
        self.t_prog=0

    def FPS_calcul(self):
        self.currentTime = time.time()      # Calculating the FPS
        self.t_prog=self.currentTime-self.previousTime
        self.FPS = 1 / (self.currentTime-self.previousTime)
        self.previousTime = self.currentTime

    def update(self):
        # Read the next frame from the stream in a different thread
        self.FPS_calcul()
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(.01)

    def show_frame(self):
        # Display frames in main program
        cv2.putText(self.frame, str(self.FPS), (10,  10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
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
        self.frame = draw_landmark(self.frame, results)
        list_joints_image = exctract_joint_image(self.frame, results)
        C="No class detected"
        if(len(list_joints_image)==42):
            list_joints_image=pre_process_landmark(list_joints_image)
            probs = predict_model_FCNN(FCNN, list_joints_image)
            C=classes[int(probs)]
            # print("Source : ",self.src, " ; Classe : ", C)
        self.max_buff = self.mean_classes(C)
        return self.max_buff





if __name__ == '__main__':
    video_stream_widget = VideoStreamWidget(0)
    # video_stream_widget1 = VideoStreamWidget(1)

    FCNN = keras.models.load_model(r'FCNN\FCNN_model')
    classes = class_extract(r'data\train')
    print("classes : ", classes)

    while True:
        try:
            video_stream_widget.detect_class(FCNN,classes)
            # video_stream_widget1.detect_class(FCNN,classes)
            
            video_stream_widget.show_frame()
            # video_stream_widget1.show_frame()
        except AttributeError:
            pass




# class camera:
#     def __init__(self, nb_cam):
#         # self.nb_cam=nb_cam
#         self.image=0
#         self.success=0
#         self.mean_C=0
#         self.cap = cv2.VideoCapture(nb_cam)
#         self.buff = collections.deque(maxlen=10)
#         self.image_height, self.image_width,_ = self.height_width()
#         if(0, _ == self.cap.read()):
#             print("ok")

#     def height_width(self):
#         _, image = self.cap.read()
#         return image.shape

#     def read_frame(self):
#         success, self.image = self.cap.read()
#         if not success:
#             print("Ignoring empty camera frame.")
#             return -1
#         return 0
        

#     def detect_class(self,FCNN,classes):
#         hands = mp_model()
#         self.image.flags.writeable = True
#         self.image, results = mediapipe_detection(self.image, hands)
#         self.image = draw_landmark(self.image, results)
#         list_joints_image = exctract_joint_image(self.image, results)
#         C="No class detected"
#         if(len(list_joints_image)==42):
#             list_joints_image=pre_process_landmark(list_joints_image)
#             probs = predict_model_FCNN(FCNN, list_joints_image)
#             C=classes[int(probs)]
#         self.mean_C = mean_classes(self.buff, C,classes)

#     def show(self):
#         cv2.putText(self.image, str(self.mean_C), (10,  self.image_height-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
#         cv2.imshow('Test class image', self.image)

    # def mean_classes(C,classes_list):
    #     buff.append(C)
    #     list_sorted =  Counter(buff)
    #     # max_key = max(list_sorted.iteritems(), key=operator.itemgetter(1))[0]
    #     max_key = max(list_sorted.items(), key=operator.itemgetter(1))[0]
    #     return max_key

# if __name__ == "__main__":
#     cap_0 = camera(0)
#     # cap_1 = camera(1)
    
#     FCNN = keras.models.load_model(r'IA_AGV\FCNN\FCNN_model')
#     classes = class_extract(r'IA_AGV\data\train')
#     print("classes : ", classes)

#     while(True):
#         cap_0.read_frame()
#         # cap_1.read_frame()
#         cap_0.detect_class(FCNN,classes)
#         # cap_1.detect_class(FCNN,classes)
#         cap_0.show()
#         # cap_1.show()
#         print("Moyenne camera 1 : ",cap_0.mean_C)#,"\n","Moyenne camera 2 : ",cap_1.mean_C)
#         if cv2.waitKey(1) & 0xFF == ord("q"): 
#             cap_0.cap.release()
#             break


