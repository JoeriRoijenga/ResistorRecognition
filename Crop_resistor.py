import cv2
import numpy as np
import time
import os, os.path
run = 0


# Stack images in one window (mostly for debugging)
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


# Function getContours
def getContours(img, orig):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:                                                               # Offset for wrong readings
            print(area)
            cv2.drawContours(imgContour, [cnt], 0, (255, 0, 0), 3)                    # Draw contour
            peri = cv2.arcLength(cnt, False)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, False)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            if w >= 400 and h >= 30:                                                  # Offset for contour mismatch
                cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)     # Show rectangle around contour
                x = x + 230                                                           # Crop only middle part of resistor
                w = w - 460                                                           # " "

                imgCropped = orig[y:y + h, x:x + w]                                   # Crop resistor
                imgCropped = cv2.resize(imgCropped, cropsize, 0, 0, cv2.INTER_AREA)   # Resize crop to 150, 50 pixels

                cv2.imshow("Crop", imgCropped)                                        # Show cropped image
                time.sleep(0.5)                                                       # Sleep for stability
                global run                                                            # Counter for how many samples
                run = run + 1
                print(run)
                cv2.imwrite("./crop/" + str(run) + ".jpg", imgCropped)                # Save cropped image


# Init camera
frameWidth = 1920
frameHeight = 1080
cap = cv2.VideoCapture(0)
cap.set(16, frameWidth)
cap.set(9, frameHeight)

# Output
cropsize = (150, 50)

# Save output
Dir = "./crop/"
filecount = next(os.walk(Dir))[2]
run = (len(filecount) - 1)


# Main
while True:
    success, img = cap.read()                                               # Open camera
    imgContour = img.copy()                                                 # Copy image
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                         # Gray filter
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), cv2.BORDER_REPLICATE)       # Blur filter
    thresh = cv2.threshold(imgBlur, 200, 255, cv2.THRESH_BINARY)[1]         # Threshhold for glare
    thresh = cv2.erode(thresh, None, iterations=5)                          # Erode filter
    thresh = cv2.dilate(thresh, None, iterations=5)                         # Dilate filter
    imgCanny = cv2.Canny(thresh, 50, 50)                                    # Canny filter

    getContours(imgCanny, img)                                              # Call function getContours
    imgStack = stackImages(1, ([img, imgCanny, imgContour]))                # Stack input, cannyfilter and contours
    cv2.imshow("Stack", imgStack)                                           # Show stack in one window

    if cv2.waitKey(1) & 0xFF == ord('q'):                                   # Show stack until next image or 'q'
        break
