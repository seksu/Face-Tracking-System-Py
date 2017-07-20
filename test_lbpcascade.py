import numpy as np
import cv2
from time import gmtime
# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

minCascade = 1.05
maxCascade = 6
countIm    = 0
prevTime   = 0
fps        = 0
face_cascade = cv2.CascadeClassifier('haarcascades_cuda/haarcascade_frontalface_default.xml') #frontalface_default
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while 1:
    ret, img = cap.read()
    countIm += 1
    currentTime = gmtime().tm_sec
    if prevTime != currentTime :
        fps = countIm
        countIm = 0 #
        prevTime = currentTime
    cv2.putText(img,str(fps),(0,20), font,0.8,(0,255,0),2)
    cv2.putText(img,'fps',(30,20), font,0.8,(0,255,0),2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, minCascade, maxCascade) #1.3 5
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'face',(x-3,y-3), font, w*h/60000,(255,0,0),2)
       
    cv2.imshow('img',img)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()