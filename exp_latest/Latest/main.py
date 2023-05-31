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
rgb_values = {}
listofPeople = []
box_id_counter = 0


def postprocess(frame, outs):
    global first_detection, rgb_values, box_id_counter, listofPeople

    boxes, confidences, classIds = detect(frame, outs)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    ppl_cntr = 0
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
        ppl_cntr += 1

        bbox = (left, top, left + width, top + height)
        used_boxes.append(bbox)

        # Calculate the part height based on the bounding box height
        part_height = height // 3  # Divide the bounding box height into 3 parts

        # Create a new person instance and store RGB values
        person = pepd.Person(bbox)
        for j in range(1, 4):
            part_top = top + ((j - 1) * part_height)
            part_bottom = part_top + part_height
            part_bbox = (left, part_top, left + width, part_bottom)
            part_frame = frame[part_top:part_bottom, left:left + width]
            avg_rgb = np.mean(part_frame, axis=(0, 1))
            person.rgb_values.append(avg_rgb)
            # ...

        # Add the person object to the list
        listofPeople.append(person)

        # ...
    return used_boxes

def print_box_info(box_id, avg_rgb):
    print("Box ID: {}".format(box_id))
    print("RGB values for each part:")
    for j in range(1, 4):
        print("Part {}: {}".format(j, avg_rgb))

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    print("Drawing boxes.")
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 2)
    print("Displaying labels")


def tracklines(used_boxes):
    global object_id_list
    global centroid_dict
    global enter_count
    global enter
    global exit
    global exit_count
    objects = tracker.update(used_boxes)
    for (objectId, newbox) in objects.items():
        x1, y1, x2, y2 = newbox
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cX = int((x1 + x2) / 2.0)
        cY = int((y1 + y2) / 2.0)
        cv2.circle(frame, (cX, cY), 4, (0, 255, 0), -1)

        centroid_dict[objectId].append((cX, cY))

        if objectId not in object_id_list:
            object_id_list.append(objectId)
            start_pt = (cX, cY)
            end_pt = (cX, cY)
            cv2.line(frame, start_pt, end_pt, (255, 0, 0), 4)

        else:
            l = len(centroid_dict[objectId])
            for pt in range(len(centroid_dict[objectId])):
                if not pt + 1 == l:
                    start_pt = (centroid_dict[objectId][pt]
                                [0], centroid_dict[objectId][pt][1])
                    end_pt = (centroid_dict[objectId][pt + 1]
                              [0], centroid_dict[objectId][pt + 1][1])
                    cv2.line(frame, start_pt, end_pt, (0, 255, 0), 4)

        if enter and exit is not None:
            enter_count, exit_count = countPeople(
                frame, enter, exit, newbox, objectId)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = "ID: {}".format(objectId)
        cv2.putText(frame, text, (x1, y1-5),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    cv2.putText(frame, "Enters: " + str(enter_count), (0, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, "Exits: " + str(exit_count), (0, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, "# inside " + str(enter_count - exit_count),
                (0, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def selectEntrance(frame):
    # Create resizable window and set properties
    cv2.namedWindow('Select the entrance', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Select the entrance',
                          cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Select entrance ROI
    enter = cv2.selectROI('Select the entrance', frame)

    # Destroy the entrance window
    cv2.destroyAllWindows()

    # Create resizable window and set properties
    cv2.namedWindow('Select the exit', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        'Select the exit', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Select exit ROI
    exit = cv2.selectROI('Select the exit', frame)

    # Destroy the exit window
    cv2.destroyAllWindows()

    return enter, exit


def countPeople(frame, enter, exit, rect, objectId):
    global enter_count
    global exit_count
    global counted_object_id
    global outed_object_id
    rx1, ry1, rx2, ry2 = rect
    enx1, eny1, enx2, eny2 = enter
    ex1, ey1, ex2, ey2 = exit

    rx1 = int(rx1)
    ry1 = int(ry1)
    rx2 = int(rx2)
    ry2 = int(ry2)

    enx1 = int(enx1)
    eny1 = int(eny1)
    enx2 = int(enx2) + enx1
    eny2 = int(eny2) + eny1

    ex1 = int(ex1)
    ey1 = int(ey1)
    ex2 = int(ex2) + ex1
    ey2 = int(ey2) + ey1

   # r1 inside r2

    if rx1 > enx1 and ry1 > eny1 and rx2 < enx2 and ry2 < eny2 and objectId not in counted_object_id:
        enter_count += 1
        counted_object_id.append(objectId)
        print(objectId, "   Entered, +")
    if rx1 > ex1 and ry1 > ey1 and rx2 < ex2 and ry2 < ey2 and objectId not in outed_object_id:
        exit_count += 1
        outed_object_id.append(objectId)
        print(objectId, "Exit, -")


    return enter_count, exit_count


more_frame = False
# Read first frame
success, frame = cap.read()

frame_count = 0
frame_interval = 1

# quit if unable to read the video file
if not success:
    print('Failed to read video')
    sys.exit(1)

while cv2.waitKey(1) < 0:

    # get frame from the video
    hasFrame, frame = cap.read()
    frame_count += 1

    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv2.waitKey(3000)
        # Release device
        cap.release()
        break

    if frame_count % frame_interval == 0:
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(
            frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        used_boxes = postprocess(frame, outs)
        tracklines(used_boxes)
        while exit is None:
            enter, exit = selectEntrance(frame)
        cv2.rectangle(frame, (enter[0], enter[1]), (enter[0] +
                      enter[2], enter[1] + enter[3]), (255, 0, 0), 2)
        cv2.rectangle(frame, (exit[0], exit[1]), (exit[0] +
                      exit[2], exit[1] + exit[3]), (255, 0, 0), 2)

        # Write the frame with the detection boxes
        vid_writer.write(frame.astype(np.uint8))
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            "Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Tracking", frame)

cap.release()
vid_writer.release()
cv2.destroyAllWindows()
