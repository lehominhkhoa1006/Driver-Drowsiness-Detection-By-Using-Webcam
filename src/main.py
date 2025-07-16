from gpiozero import Buzzer 
import numpy as np 
 
buzzer = Buzzer(24) 
 
def euclideanDistance(pointA, pointB): 
    return np.linalg.norm(pointA - pointB) 
 
def calculateEyeAspectRatio(eyePoints): 
    verticalDistance1 = euclideanDistance(eyePoints[1], eyePoints[5]) 
    verticalDistance2 = euclideanDistance(eyePoints[2], eyePoints[4]) 
    horizontalDistance = euclideanDistance(eyePoints[0], eyePoints[3]) 
    aspectRatio = (verticalDistance1 + verticalDistance2) / (2.0 * horizontalDistance) 
    return aspectRatio 
 
eyeAspectRatioThreshold = 0.25 
maxConsecutiveClosedFrames = 25 
closedEyeFrameCount = 0 
isAlarmOn = False 
isBuzzerOn = False

import cv2 
import dlib 
from imutils import face_utils 
 
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
landmarkPredictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 
 
(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] 
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

from imutils.video import VideoStream 
import imutils 
import time 
 
videoStream = VideoStream(src=0).start() 
time.sleep(1.0)

while True: 
    frame = videoStream.read() 
    frame = imutils.resize(frame, width=450) 
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
 
    faces = faceCascade.detectMultiScale(grayFrame, scaleFactor=1.1, 
        minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE) 
 
    alarmTriggered = False 
 
    for (x, y, w, h) in faces: 
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h)) 
        landmarks = landmarkPredictor(grayFrame, rect) 
        landmarks = face_utils.shape_to_np(landmarks) 
 
        leftEyePoints = landmarks[leftEyeStart:leftEyeEnd] 
        rightEyePoints = landmarks[rightEyeStart:rightEyeEnd] 
 
        leftEyeRatio = calculateEyeAspectRatio(leftEyePoints) 
        rightEyeRatio = calculateEyeAspectRatio(rightEyePoints) 
        averageEyeRatio = (leftEyeRatio + rightEyeRatio) / 2.0 
 
        leftEyeHull = cv2.convexHull(leftEyePoints) 
        rightEyeHull = cv2.convexHull(rightEyePoints) 
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1) 
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1) 
 
        if averageEyeRatio < eyeAspectRatioThreshold: 
            closedEyeFrameCount += 1 
            if closedEyeFrameCount >= maxConsecutiveClosedFrames and not alarmTriggered: 
                alarmTriggered = True 
                isAlarmOn = True 
                if not isBuzzerOn: 
                    buzzer.on() 
                    isBuzzerOn = True 
                cv2.putText(frame, "STOP!!!", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
        else: 
            closedEyeFrameCount = 0 
            isAlarmOn = False 
            alarmTriggered = False 
            normalFrameCount += 1 
            if normalFrameCount >= normalFrameThreshold: 
                normalFrameCount = 0 
                if isBuzzerOn: 
                    buzzer.off() 
                    isBuzzerOn = False 
 
        cv2.putText(frame, "EYE AVG RATIO: {:.3f}".format(averageEyeRatio), 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2) 
 
    cv2.imshow("Camera", frame) 
    key = cv2.waitKey(1) & 0xFF 
    if key == 27: 
        break 
 
cv2.destroyAllWindows() 
videoStream.stop()
