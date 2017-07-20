import numpy as np
import cv2
from time import gmtime
# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

minCascade = 1.03
maxCascade = 6
countIm    = 0
prevTime   = 0
fps        = 0
Id         = 0
score      = 0
face_cascade = cv2.CascadeClassifier('haarcascades_cuda/haarcascade_frontalface_alt.xml') #frontalface_default
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load('trainer/trainer2.yml')

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
        Id, score = recognizer.predict(gray[y:y+h,x:x+w])

        if(Id == 1 and score < 50) :
            cv2.putText(img,'Pare',(x-3,y-3), font, w*h/60000,(255,0,0),2)
        
        else :
            cv2.putText(img,'Unknow',(x-3,y-3), font, w*h/60000,(255,0,0),2)
        
        #cv2.putText(img,'face',(x-3,y-3), font, w*h/60000,(255,0,0),2)
    cv2.putText(img,str(Id),(0,40), font,0.5,(0,255,0),2)
    cv2.putText(img,str(score),(0,55), font,0.5,(0,255,0),2)   
    cv2.imshow('img',img)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()