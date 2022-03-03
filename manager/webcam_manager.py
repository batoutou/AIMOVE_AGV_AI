import cv2
import time

def FPS(classe,previousTime, image):
    image=cv2.flip(image, 1)       # Flip the image horizontally for a selfie-view display.
    currentTime = time.time()      # Calculating the FPS
    fps = 1 / (currentTime-previousTime)
    previousTime = currentTime
    cv2.putText(image, str(int(fps))+" FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)  # Displaying FPS on the image
    cv2.putText(image, str("Classe : "+classe), (10, 140), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2) 
    return previousTime, image

# def classe_display(classe, image):
#     cv2.putText(image, str("Classe : "+classe), (10, 140), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)  # Displaying FPS on the image
#     return cv2.flip(image, 1)