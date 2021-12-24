import cv2
import mediapipe as mp
import numpy as np
import time


class handDetector():

    # Firstly the initialize constructor is already defined if need be you can make any changes to this constructor
    def __init__(self, mode=False, maxHands = 2, detectionCon = 0.7, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

#############################################################################################################
# findHands function takes the image source from the calling block and if the input draw = True then draw the
# hands with all the landmarks using mediapipe Hands solution (Read the doc for details)
# Also add the results to the class variable results which would then be used for further calculations

    def findHands(self, img, draw=True):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if draw and self.results.multi_hand_landmarks:
            for lm in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, lm, self.mpHands.HAND_CONNECTIONS)
        return img


# findPosition function takes the image source, hand we are currently working in the image source
# and if to draw or not from the calling block and if the input draw = True then draw the
# hand positions with all the landmarks using mediapipe Hands solution (Read the doc for details)
# also make the x-coordinate, y-coordinate and the the rectangle containing the hand and also the landmark list
# Also add the results to the class variable results which would then be used for further calculations
    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img)
        x1, x2, x3, x4, y1, y2, y3, y4 = (0, 0, 0, 0, 0, 0, 0, 0)
        if self.results.multi_hand_landmarks:  # returns None if hand is not found
            hand = self.results.multi_hand_landmarks[handNo]
            for lM in hand.landmark:
                (h, w, c) = img.shape
                xList.append(int(lM.x * w))
                yList.append(int(lM.y * h))
            x1, y1 = min(xList) - 30, min(yList) - 30
            x2, y2 = max(xList) + 30, y1
            x3, y3 = x2, max(yList) + 30
            x4, y4 = x1, y3
            bbox = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

        # Draw if the draw given is true
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if self.results.multi_hand_landmarks and draw:
            for idx, xyz in enumerate(self.results.multi_hand_landmarks[handNo].landmark):
                self.lmList.append([idx, xyz.x * img.shape[1], xyz.y * img.shape[0]])

        return self.lmList, bbox

# findDistance function returns the image after drawing distance between 2 points
# and drawing thar distance and highlighting the points with r radius circle and
# t thickness line and also return length there
# this function would help us make our click to execute

    def findDistance(self,  p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
        if draw:
            cv2.circle(img, (int(x1), int(y1)), r, (0, 0, 255), 1)
            cv2.circle(img, (int(x2), int(y2)), r, (0, 0, 255), 1)
            cv2.line(img, (int(x2), int(y2)), (int(x1), int(y1)), (0, 0, 255), t)

        length = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        return length, img, [x1, y1, x2, y2, cx, cy]

# fingersUp function return list of 5 fingers and their respective state
# 0- down and 1- Up
# Make sure to go through the mediapipe docs to get to know landmark
# number of each finger and a method to know if the finger is up or not

    def fingersUp(self):
        tipIds = [4, 8, 12, 16, 20]
        fingers = [0, 0, 0, 0, 0]
        if self.lmList[17][1] <= self.lmList[5][1]:
            if self.lmList[tipIds[0]][1] <= self.lmList[5][1]:
                fingers[0] = 0
            else:
                fingers[0] = 1
        else:
            if self.lmList[tipIds[0]][1] <= self.lmList[5][1]:
                fingers[0] = 1
            else:
                fingers[0] = 0
        for idx in range(1, 5):
            if self.lmList[tipIds[idx]][2] <= self.lmList[tipIds[idx] - 2][2]:
                fingers[idx] = 1
            else:
                fingers[idx] = 0
        return fingers

#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################





# Now for the main function is to check and debug the class
# You may change it any way you want
# I have added the FPS counter and take the video feed from the PC
# If you do not have a webcam in your PC you can use DROID CAM Software
# To debug you can also use image of a hand , the code for this I have commented out
# you can de-comment it out and comment the video feed code to debug if you feel
# some function is not working as required


#############################################################################################################

def main():
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0)
    detector = handDetector(maxHands=1)

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if bbox:
            cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 3)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

        # length, img, [x1, y1, x2, y2, cx, cy] = 0, img, [0, 0, 0, 0, 0, 0]
        #
        # if lmList:
        #     length, img, [x1, y1, x2, y2, cx, cy] = detector.findDistance(4, 20, img)
        if lmList:
            fingers = detector.fingersUp()
            for n, i in enumerate(fingers):
                cv2.putText(img, 'finger ' + str(n+1) + ': ' + str(i), (10, 80 + 30*n), cv2.FONT_HERSHEY_PLAIN, 2,
                            (0, 0, 0), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()