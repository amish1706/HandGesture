import HandTrackingModule as htm
import cv2
import numpy as np
import autopy
import mouse
import time

wCam, hCam = 1280, 720
frameR = 40
smoothening = 2

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()

while True:
    _, img = cap.read()
    # Find Hand LandMarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # Get tip of index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # Check which finger is up
        fingers = detector.fingersUp()
        print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam-frameR, hCam-frameR), (255,0,255),2)
        # Only index finger : moving mode
        if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            # Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            # Smoothen Values
            clocX = plocX + (x3 - plocX)/smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            # Move Mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (int(x1), int(y1)), 15, (0,255,0), cv2.FILLED)
            plocX, plocY = clocX, clocY
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img, draw=False)
            if fingers[3] == 1 and fingers[4] == 0:
                mouse.click('right')
                cv2.putText(img, "Right Click", (15, 40), 1, 1, (255, 0, 0), 3)
            if length < 65:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), -1)
                mouse.click('left')
                cv2.putText(img, "Left Click", (15, 40), 1, 1, (255, 0, 0), 3)
        if fingers[0] == 1 and fingers[1] == 1:
            length, _, _ = detector.findDistance(8, 4, img, draw=False)
            if length < 50:
                autopy.mouse.toggle(down=True)
            if length > 50:
                autopy.mouse.toggle(down=False)
        if fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            mouse.wheel(-1)
            cv2.putText(img, "Scroll Down", (15, 40), 1, 1, (255, 0, 0), 3)
        elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
            mouse.wheel(1)
            cv2.putText(img, "Scroll Up", (15, 40), 1, 1, (255, 0, 0), 3)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (15, 15), 1, 1, (255, 0, 0), 3)
    cv2.resize(img, (640, 360))
    cv2.imshow("Image", img)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()






