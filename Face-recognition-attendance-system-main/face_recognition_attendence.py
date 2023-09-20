import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ReferenceImages'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
 
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #A list of 128-dimensional face encodings (one for each face in the image)
        encode = face_recognition.face_encodings(img)[0]     
        encodeList.append(encode)
    return encodeList


def markAttendence(name):
    with open('Attendence.csv','r+') as f:
        myDatalist = f.readlines()
        namelist = []
        for line in myDatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtString = now.strftime("%H-%M:%S")
            f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images)
print('Encoding Completed!')


cap = cv2.VideoCapture(0)

while True:
    sucess ,img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    #A list of tuples of found face locations in css (top, right, bottom, left) order
    faces_in_current_frame = face_recognition.face_locations(imgS)  

    #A list of 128-dimensional face encodings (one for each face in the image)
    encodes_current_frame = face_recognition.face_encodings(imgS,faces_in_current_frame)    

    for encodeFaces , faceloc in zip(encodes_current_frame,faces_in_current_frame):
        #A list of True/False values indicating which known_face_encodings match the face encoding to check
        matches = face_recognition.compare_faces(encodeListKnown,encodeFaces)   
        #A numpy ndarray with the distance for each face in the same order as the ‘faces’ array
        facedis = face_recognition.face_distance(encodeListKnown,encodeFaces)   
        #print(facedis)

        matchIndex =np.argmin(facedis)

        if matches[matchIndex]:
            #Prints the name of the face in the camera
            name = classNames[matchIndex].upper()  
            # print(name)

            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(39,157,154),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(39,154,157),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendence(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)