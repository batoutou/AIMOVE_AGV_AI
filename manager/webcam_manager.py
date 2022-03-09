import cv2
import time

def FPS(message, classe,previousTime, image):
    image=cv2.flip(image, 1)       # Flip the image horizontally for a selfie-view display.
    currentTime = time.time()      # Calculating the FPS
    t_prog=currentTime-previousTime
    fps = 1 / (currentTime-previousTime)
    previousTime = currentTime
    cv2.putText(image, str(int(fps))+" FPS", (10, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)  # Displaying FPS on the image
    # cv2.putText(image, str("Classe : "+classe), (10, 180), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv2.putText(image, str(message), (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2) 
    return t_prog, previousTime, image

# def classe_display(classe, image):
#     cv2.putText(image, str("Classe : "+classe), (10, 140), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)  # Displaying FPS on the image
#     return cv2.flip(image, 1)

def sleep(time_prog, FPS_selected):
    if(time_prog < 1/FPS_selected):
        # time.sleep(2*1/FPS_selected- time_prog)
        time.sleep(1/FPS_selected- time_prog)