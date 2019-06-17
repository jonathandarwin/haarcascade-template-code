import cv2
import numpy as np

# Load model XML
classfiers = []
classifiers.append(cv2.CascadeClassifier('haarcascade_fire_weapon.xml'))
classifiers.append(cv2.CascadeClassifier('haarcascade_weapon.xml'))
classifiers.append(cv2.CascadeClassifier('haarcascade_drugs.xml'))
classifiers.append(cv2.CascadeClassifier('haarcascade_bomb.xml'))

# Template code ini menggunakan webcam sebagai pengganti scanner
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()    
    frame = cv2.flip(frame, 1)

    for classifier in classifiers:
        objectDetected = classifier.detectMultiScale(frame, 1.3, 5)
        if objectDetected is not ():
            for (x,y,w,h) in objectDetected:        
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255),2)
                cv2.putText(frame, 'Object Found', (100,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)            
    
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()