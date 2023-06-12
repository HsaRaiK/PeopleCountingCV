from __future__ import print_function
import sys
import cv2
import numpy as np
from random import randint
from tkinter import Tk     # pip install tk
from tkinter.filedialog import askopenfilename
from collections import defaultdict
import argparse
import ppl_database as pepd

import guiv1k1 as guiv1
from methods import *


centroid_dict = defaultdict(list)
rgb_values = {}


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]


def detect(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Calculate the center of the screen
    centerX = int(frameWidth / 2)
    centerY = int(frameHeight / 2)

    # Calculate the limits of the detection area
    limitX = int(frameWidth * 0.3)
    limitY = int(frameHeight)

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []

    isDetected = False
    for out in outs:
        for detection in out:
            if isDetected == False:
                isDetected = True
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
                if classId == 0 and abs(centerX - center_x) < limitX and abs(centerY - center_y) < limitY:
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

    return boxes, confidences, classIds


first_detection = True
listofPeople = []

def unified_process(frame, outs):
    global first_detection, listofPeople, centroid_dict, rgb_values

    boxes, confidences, classIds = detect(frame, outs)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    used_boxes = []

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

        bbox = (left, top, left + width, top + height)
        used_boxes.append(bbox)

        if first_detection:
            listofPeople.append({'id': i, 'box': bbox})
            rgb_values[i] = (randint(0, 255), randint(0, 255), randint(0, 255))
        else:
            # Check if the box ID already exists in the listofPeople
            if any(person['id'] == i for person in listofPeople):
                # Update the box coordinates for the existing ID
                person = next(person for person in listofPeople if person['id'] == i)
                person['box'] = bbox
            else:
                # Add a new box ID to the listofPeople
                listofPeople.append({'id': i, 'box': bbox})
                rgb_values[i] = (randint(0, 255), randint(0, 255), randint(0, 255))

        label = f"Person {i}"
        cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 0, 255), 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Calculate the centroid of the bounding box
        centroid_x = (left + left + width) / 2
        centroid_y = (top + top + height) / 2
        centroid_dict[i].append((int(centroid_x), int(centroid_y)))

    if not first_detection:
        # Track the movement of each person
        for person in listofPeople:
            points = centroid_dict[person['id']]
            if len(points) > 1:
                # Draw the trajectory of the person
                for j in range(1, len(points)):
                    prev_point = points[j - 1]
                    curr_point = points[j]
                    cv2.line(frame, prev_point, curr_point, rgb_values[person['id']], 2)

    first_detection = False

    # Remove the bounding boxes that are no longer in use
    listofPeople[:] = [person for person in listofPeople if person['box'] in used_boxes]

    return frame


# Load the configuration and weights files for the model
configPath = 'yolov3.cfg'
weightsPath = 'yolov3.weights'

# Load the network
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get the output layer names
outNames = getOutputsNames(net)

# Set the confidence threshold and non-maximum suppression threshold
confThreshold = 0.5
nmsThreshold = 0.4

# Open the video file or capture device
Tk().withdraw()
filename = askopenfilename()
cap = cv2.VideoCapture(filename)

while True:
    # Read a frame
    hasFrame, frame = cap.read()

    if not hasFrame:
        break

    # Create a 4D blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)

    # Set the input to the network
    net.setInput(blob)

    # Forward pass through the network
    outs = net.forward(outNames)

    # Perform object detection and tracking
    frame = unified_process(frame, outs)

    # Display the resulting frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video file or capture device
cap.release()

# Close all windows
cv2.destroyAllWindows()
