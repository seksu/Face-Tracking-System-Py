import numpy as np
import cv2
from time import gmtime

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

minCascade = 1.06   # set minimal cascade
maxCascade = 5      # set maximal cascade
countIm    = 0
prevTime   = 0
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml') #frontalface_default
eye_cascade  = cv2.CascadeClassifier('haarcascades/haarcascade_eye_tree_eyeglasses.xml')
body_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_fullbody.xml') 
upperbody_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_upperbody.xml')
lowerbody_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_lowerbody.xml')
font = cv2.FONT_HERSHEY_SIMPLEX # font for print text on picture

cap = cv2.VideoCapture(0)   #create object for camera 0 is default webcam

while 1:
    ret, img = cap.read()   #capture image from webcam
    countIm += 1
    currentTime = gmtime().tm_sec
    if prevTime != currentTime :
        fps = countIm
        countIm = 0 #
        prevTime = currentTime
    cv2.putText(img,str(fps),(0,20), font,0.8,(0,255,0),2)
    cv2.putText(img,'fps',(30,20), font,0.8,(0,255,0),2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray',gray)
    faces = face_cascade.detectMultiScale(gray, minCascade, maxCascade) #1.3 5
    bodys = body_cascade.detectMultiScale(gray, minCascade, maxCascade)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'face',(x-3,y-3), font, w*h/60000,(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, minCascade, maxCascade)
        for (ex,ey,ew,eh) in eyes:
            cv2.putText(roi_color,'eye',(ex-3,ey-3), font, ew*eh/6000,(0,255,0),2)
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    
    for (bx,by,bw,bh) in bodys :
        cv2.rectangle(img,(bx,by),(bx+bw,by+bh),(0,0,255),2)
        cv2.putText(img,'body',(bx-3,by-3), font, bw*bh/60000,(0,0,255),2)
        bodyRoi_gray = gray[by:by+bh, bx:bx+bw]
        bodyRoi_color = img[by:by+bh, bx:bx+bw]

        upperbodys = upperbody_cascade.detectMultiScale(bodyRoi_gray, minCascade, maxCascade)
        for (ubx,uby,ubw,ubh) in upperbodys :
            cv2.putText(bodyRoi_color,'upper',(ubx-3,uby-3), font, ubw*ubh/50000,(255,255,0),2)
            cv2.rectangle(bodyRoi_color,(ubx,uby),(ubx+ubw,uby+ubh),(255,255,0),2)

        lowerbodys = lowerbody_cascade.detectMultiScale(bodyRoi_gray, minCascade, maxCascade)
        for (lbx,lby,lbw,lbh) in lowerbodys :
            cv2.putText(bodyRoi_color,'lower',(lbx-3,lby-3), font, lbw*lbh/60000,(255,255,0),2)
            cv2.rectangle(bodyRoi_color,(lbx,lby),(lbx+lbw,lby+lbh),(255,255,0),2)
    
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()   