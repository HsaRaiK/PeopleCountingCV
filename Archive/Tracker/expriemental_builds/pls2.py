from __future__ import print_function
import sys
import cv2
import numpy as np
from random import randint
from tkinter import Tk     # pip install tk
from tkinter.filedialog import askopenfilename

import argparse


confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image


def createTracker():
    tracker = cv2.legacy.TrackerKCF_create()
    return tracker

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]
    
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
 
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    ppl_cntr = 0
    for out in outs:
        for detection in out:
            print("Detecting...")
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
 
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        
        if classIds[i] == 0:
    	    ppl_cntr += 1
    	    #drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
        multiTracker.add(createTracker(), frame, boxes[i])
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 255, 0), 2, 1)
    cv2.putText(frame, str(ppl_cntr), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)      
    return boxes    
# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    print("Drawing boxes.")
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 2)
    print("Displaying labels")


classesFile = "coco.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

 
# Path to the input video file
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
VIDEO_PATH = askopenfilename()
multiTracker = cv2.legacy.MultiTracker_create()
cap = cv2.VideoCapture(VIDEO_PATH)
outputFile = VIDEO_PATH[:-4]+'_out.avi'
print("Generated empty output.")
vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Read first frame
success, frame = cap.read()
# quit if unable to read the video file
if not success:
  print('Failed to read video')
  sys.exit(1)

while cv2.waitKey(1) < 0:
     
    # get frame from the video
    hasFrame, frame = cap.read()
    print("Getting frames.")
    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv2.waitKey(3000)
        # Release device
        cap.release()
        break
 
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
 
    # Sets the input to the network
    net.setInput(blob)
 
    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))
 
    # Remove the bounding boxes with low confidence
    boxes = postprocess(frame, outs)
 
    # Write the frame with the detection boxes
    vid_writer.write(frame.astype(np.uint8))





