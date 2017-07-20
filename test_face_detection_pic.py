import cv2
import numpy as np

minCascade      = 1.03
maxCascade      = 6
name 			= 1
i 				= 0
offset          = 50
img 			= cv2.imread('pic/pare.jpg')
face_cascade    = cv2.CascadeClassifier('lbpcascades/lbpcascade_frontalface.xml') #frontalface_default
font = cv2.FONT_HERSHEY_SIMPLEX

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, minCascade, maxCascade) #1.3 5
for (x,y,w,h) in faces:
	cv2.imwrite("capture/face-"+str(name)+'.'+ str(i) + ".jpg", gray[y-offset:y+h+offset,x-offset:x+w+offset])
	i+=1
	cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	cv2.putText(img,'face',(x-3,y-3), font, w*h/60000,(255,0,0),2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()