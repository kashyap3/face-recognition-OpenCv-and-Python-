import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time
# from PIL import ImageGrab
img_counter = 0
path = 'E:/projects/face-recognition/ImagesRecord'
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markRecord(name):
    with open('unknown.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%m/%d/%Y, %H:%M:%S")
            f.writelines(f'\n{name},{dtString}')
        #else:
         #   now = datetime.now()
          #  dtString = now.strftime('%H:%M:%S')
           # f.writelines()


#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:

    success, img = cap.read()
    # img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex] < 0.50:
            name = classNames[matchIndex].upper()
            markRecord(name)
        else:

            cv2.namedWindow("test")



            while True:
                frame_rate = cap.get(5)
                ret, frame = cap.read()
                if not ret:
                    print("failed to grab frame")
                    break
                cv2.imshow("test", frame)

                k = cv2.waitKey(1)
                img_name = "OPENCV_FRAME_{}.jpg".format(img_counter)
                markRecord(img_name)

                hello = 'E:/projects/face-recognition/ImagesRecord/' + img_name
                cv2.imwrite(hello, frame)
                print("{} written!".format(img_name))
                #hello = cv2.cvtColor(hello, cv2.COLOR_BGR2RGB)
                #encodeListKnown.append(findEncodings(hello))
                img_counter += 1
                time.sleep(5)
                break




cv2.imshow('Webcam', img)
cv2.waitKey(1)