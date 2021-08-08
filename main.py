import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path ='Identity'

images =[]
names =[]

myList = os.listdir(path)

def Attendence(name):
    with open('AttendenceSheet.csv', 'r+') as f:
        Present = f.readlines()
        PresentNames = []
        for i in Present:
            entry = i.split(',')
            PresentNames.append(entry[0])
        if name not in PresentNames:
            now = datetime.now()
            time = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{time}')



for person in myList:
    currentPerson = cv2.imread(f'{path}/{person}')
    images.append(currentPerson)
    names.append(os.path.splitext(person)[0])


encodeList =[]
for img in images:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodeList.append(face_recognition.face_encodings(img)[0])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (0,0),fx=0.5,fy=0.5)
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    currentFaceLocation = face_recognition.face_locations(frame2)
    currentEncode = face_recognition.face_encodings(frame2,currentFaceLocation)

    for encodeFace,locationFace in zip(currentEncode,currentFaceLocation):
        result = face_recognition.compare_faces(encodeList, encodeFace)
        faceDistance = face_recognition.face_distance(encodeList,encodeFace)
        print(faceDistance)
        match = np.argmin(faceDistance)

        if result[match]:
            name = names[match]
            print(name)
            y1,x2,y2,x1 = locationFace    
            cv2.rectangle(frame, (x1, y1), (x2,y2),(255,0,0),1)    
            cv2.rectangle(frame,(x1, y2-30),(x2,y2),(255,0,0),cv2.FILLED)            
            cv2.putText(frame, name, (x1+6,y1-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
            Attendence(name)


    cv2.imshow('Screen', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()