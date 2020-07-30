import cv2,os
import numpy as np
from PIL import Image 
import pickle
import pyttsx3
import time


recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('trainer/trainer.yml')
cascadePath = "Classifiers/face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
path = 'dataSet'

cam = cv2.VideoCapture(0)

count=0
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 255, 255)
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])        
        print (recognizer.predict(gray[y:y+h,x:x+w]))
        print ('nbr val',nbr_predicted)
        print ('confi' , conf)
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        if(nbr_predicted==1 and conf>50):
             nbr_predicted='jinu mohan'
             print ('jinumohan detected')
             count=count+1
             if(count==10):
                 engine = pyttsx3.init()
                 engine.say("hi jinu nice to meet you ")
                 engine.runAndWait() 
             

        cv2.putText(im, str(nbr_predicted)+"--"+str(conf), (x,y+h), fontface, fontscale, fontcolor) #cv2.PutText(cv2.fromarray(im),str(nbr_predicted)+"--"+str(conf), (x,y+h),font, 255) #Draw the text
    cv2.imshow('im',im)
    if cv2.waitKey(10) == ord('q'):
        break


# Stop the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()





