#    Copyright (c) 2022
#    Author      : Bruno Capuano
#    Create Time : 2022 June
#    Change Log  :
#    – Load and image and remove the background
#    – Show the original image and the background removed image
#
#    The MIT License (MIT)
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in
#    all copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#    THE SOFTWARE.

import os
import re
import math
import time
import json
import cv2
import numpy as np
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from pathlib import Path
from enum import Enum

class PositionEnum(Enum):
    POS_TL = 0
    POS_TR = 1
    POS_BL = 2
    POS_BR = 3
    POS_CENTER = 4
    POS_FACE = 5
    POS_TC = 6
    POS_FULL = 7

green = (76, 254, 116)
l_green = ((116-10)/2, 100, 100)
u_green = ((116+10)/2, 255, 255)

segmentor = SelfiSegmentation()

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# read image
camVideo = 0 #'BackgroundImages/webcam.mp4'
camCap = cv2.VideoCapture(camVideo)

#overlayVideo = 'BackgroundImages/FemaleGhost.mp4'
#overlayVideo = 'BackgroundImages/Ghost2.mp4'
#overlayVideo = 'BackgroundImages/Ghost3.mp4'

pattern = re.compile("^Ghost([0-9]+).mp4$")
overlayFiles  = [f for f in os.listdir("BackgroundImages/") if pattern.match(f)]
overlayFiles = sorted(overlayFiles)
print("Available files ",overlayFiles)
overlayIdx = 1

minSize = 30
start_frame_number = 690
camCap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
started = False
framesToSkip = 0

while camCap.isOpened():
    # Read the frame
    print("----")
    startTime = time.process_time()

    fullscreenImg = np.zeros((1080, 1920, 3), dtype = np.uint8)

    flags, camImg = camCap.read()
    if not flags:
        continue
    
    # The frame is ready and already captured
    pos_frame = camCap.get(cv2.CAP_PROP_POS_FRAMES)
    print(str(pos_frame)+" frames")
        
    #cv2.imshow('img2', camImg)
    camImg = cv2.resize(camImg, (1920, 1080))
    #print(camImg.shape)

    # Convert to grayscale
    gray = cv2.cvtColor(camImg, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray', gray)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, minSize, 1000)

    # Draw the rectangle around each face
    #print(len(faces))
    for (x, y, w, h) in faces:
        #print(w, h)
        cv2.rectangle(camImg, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display
    #cv2.imshow('img', camImg)

    if cv2.waitKey(1) & 0xFF == ord('q'): #  by press 'q' you quit the process
        break
    
    #continue

    if (len(faces) > 0) or started:
        if (len(faces) > 0):
            facePos = faces[0]
            faceCol = 1920 - math.trunc(facePos[0]+(facePos[2] / 2))
            faceRow = math.trunc(facePos[1]+(facePos[3] / 2))

        if (not started):
            overlayVideoFilename = "BackgroundImages/"+overlayFiles[overlayIdx]
            overlayMetadataFilename = Path(overlayVideoFilename).with_suffix(".json")

            overlayCap = cv2.VideoCapture(overlayVideoFilename)
            overlayMetaFile = open(overlayMetadataFilename)
            overlayMetadata = json.load(overlayMetaFile)
            overlayMetaFile.close()

            preferredPos = PositionEnum(overlayMetadata["preferredPosition"])
            masked = overlayMetadata["masked"]
            print("PreferredPos: "+str(preferredPos))
            print("Masked: "+str(masked))

        started = True
        
        flags, overlayImg = overlayCap.read()
        if not flags:
            print("End of overlay video")
            
            overlayCap.release()
            overlayIdx = (overlayIdx + 1) % len(overlayFiles)
            started = False
            continue

        if (preferredPos == PositionEnum.POS_FULL):
            overlayImg = cv2.resize(overlayImg, (1920, 1080))
        else:
            overlayImg = cv2.resize(overlayImg, (640, 480))
        #cv2.imshow('overlayImg', overlayImg)

        # The frame is ready and already captured
        pos_frame = overlayCap.get(cv2.CAP_PROP_POS_FRAMES)
        print(overlayVideoFilename + ": " + str(pos_frame)+" overlay frames")

        overlayCap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame+framesToSkip)

        rows,cols,channels = overlayImg.shape

        startRow = 0
        endRow = rows
        startCol = 0
        endCol = cols
        match preferredPos:
            case PositionEnum.POS_TL,PositionEnum.POS_FULL:
                startRow = 0
                startCol = 0
            case PositionEnum.POS_BL:
                startRow = 1080-rows
                startCol = 0
            case PositionEnum.POS_TR:
                startRow = 0
                startCol = 1920-cols
            case PositionEnum.POS_BR:
                startRow = 1080-rows
                startCol = 1920-cols
            case PositionEnum.POS_CENTER:
                startRow = max(0, math.trunc((1080-rows) / 2))
                startCol = max(0, math.trunc((1920-cols) / 2))
            case PositionEnum.POS_FACE:
                startRow = max(0, math.trunc(faceRow - (rows / 2)))
                startCol = max(0, math.trunc(faceCol - (cols / 2)))
            case PositionEnum.POS_TC:
                startRow = 0
                startCol = max(0, math.trunc((1920-cols) / 2))

        endRow = min(1080, startRow + rows)
        endCol = min(1920, startCol + cols)
        
        startRow = endRow - rows
        startCol = endCol - cols

        print("face: "+str(faceRow)+";"+str(faceCol))
        print("rows: "+str(startRow)+";"+str(endRow))
        print("cols: "+str(startCol)+";"+str(endCol))

        roi = camImg[startRow:endRow, startCol:endCol]

        green = overlayImg[2,2]
        #print("BGR:" , green)

        overlayImgHSV = cv2.cvtColor(overlayImg, cv2.COLOR_BGR2HSV)
        hsvGreen = overlayImgHSV[5,5]
        l_green = np.array([(hsvGreen[0]-10), 100, 100])
        u_green = np.array([(hsvGreen[0]+10), 255, 255])
        #print("HSV:" , hsvGreen, l_green, u_green)

        imgNoBg = segmentor.removeBG(camImg, green)
        #cv2.imshow('imgNoBg', imgNoBg)

        if masked:
            imgMask = cv2.inRange(imgNoBg, green, green)
            imgMask = imgMask[startRow:endRow, startCol:endCol]
            #cv2.imshow('imgMask', imgMask)

            # ghost masked by me
            imgMaskedGhost = cv2.bitwise_and(overlayImg, overlayImg, mask=imgMask)
            # get (i, j) positions of all RGB pixels that are black (i.e. [0, 0, 0])
            black_pixels = np.where(
                (imgMaskedGhost[:, :, 0] == 0) & 
                (imgMaskedGhost[:, :, 1] == 0) & 
                (imgMaskedGhost[:, :, 2] == 0)
            )

            # set those pixels to white
            imgMaskedGhost[black_pixels] = green
        else:
            # ghost masked by me
            imgMaskedGhost = overlayImg

        #cv2.imshow('imgMaskedGhost2', imgMaskedGhost)

        #imgBg = segmentor.removeBG(imgMaskedGhost, camImg)

        imgMaskedGhostHSV = cv2.cvtColor(imgMaskedGhost, cv2.COLOR_BGR2HSV)
        #print(imgMaskedGhostHSV[0,0])
        #print(imgMaskedGhost[0,0])

        imgGhostMask = cv2.inRange(imgMaskedGhostHSV, l_green, u_green)
        #cv2.imshow('imgGhostMask', imgGhostMask)
        imgGhostMaskInv = cv2.bitwise_not(imgGhostMask)

        #cv2.imshow('imgGhostMaskInv', imgGhostMaskInv)
        camImgBg = cv2.bitwise_and(roi, roi, mask = imgGhostMask)
        #cv2.imshow('camImgBg', camImgBg)
        overlayImgFg = cv2.bitwise_and(overlayImg, overlayImg, mask = imgGhostMaskInv)

        # overlayImgFg is the ghost image with black background. Show full screen
        #cv2.imshow('overlayImgFgFS', overlayImgFgFS)
        
        final = cv2.add(camImgBg, overlayImgFg)
        camImg[startRow:endRow, startCol:endCol] = final
        
        fullscreenImg[startRow:endRow, startCol:endCol] = overlayImgFg

        cv2.namedWindow('overlayImgFg', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('overlayImgFg', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # overlayImgFg is the ghost image with black background. Show full screen
        #cv2.imshow('overlayImgFg', fullscreenImg)

        endTime = time.process_time()
        elapsedTime = endTime - startTime
        fps = overlayCap.get(cv2.CAP_PROP_FPS) / 2
        framesToSkip = elapsedTime * fps
        print("elapsed ", elapsedTime, "; FPS ", fps, "; skip ", framesToSkip)

    #cv2.imshow('Final', camImg)
    cv2.imshow('overlayImgFg', fullscreenImg)

