import numpy as np
import cv2
face_casc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('ben.jpg')
resized=cv2.resize(img,(600,400))
face = face_casc.detectMultiScale(resized,scaleFactor=1.05,minNeighbors=9)
for x,y,w,h in face:
    im = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("mb",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
