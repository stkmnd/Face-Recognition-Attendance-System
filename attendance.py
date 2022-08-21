import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


def findEncodings(imageList):
    encodeList = []
    for img in imageList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        dataList = f.readlines()
        nameList = []

        for line in dataList:
            entry = line.split(',')
            nameList.append(entry[0])

        # print(nameList)
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H: %M: %S')
            f.writelines(f'{name}, {dtString}\n')
    return


path = 'attendanceImages'
myList = os.listdir(path)
images = []
names = []

for img in myList:
    currImg = cv2.imread(f'{path}/{img}')
    images.append(currImg)
    names.append(os.path.splitext(img)[0])

encodeList = findEncodings(images)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceLocations = face_recognition.face_locations(imgS)
    encodings = face_recognition.face_encodings(imgS, faceLocations)

    for encoding, faceLocation in zip(encodings, faceLocations):
        matches = face_recognition.compare_faces(encodeList, encoding)
        faceDistance = face_recognition.face_distance(encodeList, encoding)
        matchIndex = np.argmin(faceDistance)

        if matches[matchIndex]:
            name = names[matchIndex].upper()
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
