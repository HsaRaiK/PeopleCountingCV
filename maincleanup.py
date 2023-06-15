from __future__ import print_function
import sys
import cv2
import numpy as np
from random import randint
from tkinter import Tk     # pip install tk
from tkinter.filedialog import askopenfilename
from collections import defaultdict
import argparse
from CentroidTracker import CentroidTracker
import guiv1k1 as guiv1 
import person
import csv

confThreshold = 0.4  #Confidence thresholddnn
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image
ttl_cntr = 0
selected = 0
centroid_dict = defaultdict(list)
object_id_list = []
door = None

enter_count = 0
exit_count = 0
counted_object_id = []
outed_object_id = []
list_people = defaultdict(list)
person_info_file = "person_info.csv"


### Here is where the code starts

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

def detect(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Define the left and right boundaries for the central 50%
    left_boundary = frameWidth // 6
    right_boundary = frameWidth - left_boundary

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []

    isDetected = False
    for out in outs:
        for detection in out:
            if isDetected == False:
                # print("Detecting...")
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

                # Check if the box is within the central 50% on the x-axis
                if left < left_boundary or (left + width) > right_boundary:
                    continue

                if classId == 0:
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    return boxes, confidences, classIds

def postprocess(frame, outs):
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.

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
       
    cv2.putText(frame, "People counter: " + str(ppl_cntr), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255,  255), 2)
    return used_boxes
        

import numpy as np

### Detection done



def tracklines(used_boxes):
    global object_id_list
    global centroid_dict
    global enter_count
    global door
    global exit_count
    global counted_object_id
    global outed_object_id
    
    objects = tracker.update(used_boxes)
    
    for (objectId, newbox) in objects.items():
        x1, y1, x2, y2 = map(int, newbox)
        cX = (x1 + x2) // 2
        cY = (y1 + y2) // 2
        cv2.circle(frame, (cX, cY), 4, (0, 255, 0), -1)

        centroid_dict[objectId].append((cX, cY))
        indv = person.Person(objectId, (cX, cY))

        # Divide the box into 4 equal parts
        part_width = (x2 - x1) // 2
        part_height = (y2 - y1) // 2

        # Calculate histograms for each part
        histograms = []
        if indv.person_id not in list_people:
            for i in range(2):
                for j in range(2):
                    part_x1 = x1 + i * part_width
                    part_y1 = y1 + j * part_height
                    part_x2 = part_x1 + part_width
                    part_y2 = part_y1 + part_height
                    part = frame[part_y1:part_y2, part_x1:part_x2]

                    # Calculate histogram for the part
                    hist = cv2.calcHist([part], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                    hist = cv2.normalize(hist, hist).flatten()

                    histograms.append(hist)
                    indv.histograms = hist
                    list_people[indv] = [indv.person_id]
                    #print(indv.histograms)

        if objectId not in object_id_list:
            object_id_list.append(objectId)
            start_pt = (cX, cY)
            end_pt = (cX, cY)
            
        else:
            l = len(centroid_dict[objectId])
            for pt in range(len(centroid_dict[objectId])):
                if pt + 1 != l:  # Corrected the condition
                    start_pt = (centroid_dict[objectId][pt][0], centroid_dict[objectId][pt][1])
                    end_pt = (centroid_dict[objectId][pt + 1][0], centroid_dict[objectId][pt + 1][1])
                    cv2.line(frame, start_pt, end_pt, (0, 255, 0), 4)
        
        # Calculate histogram for the new person's bounding box
        # Create a mask image of the same size as the frame
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Draw a rectangle corresponding to the new person's bounding box on the mask
        x1, y1, x2, y2 = map(int, newbox)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Calculate histogram for the new person's bounding box using the mask
        new_hist = cv2.calcHist([frame], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        new_hist = cv2.normalize(new_hist, new_hist).flatten()

        # Compare the new histogram with stored histograms
        best_match = None
        best_score = 0.0

        for hist in histograms:
            score = cv2.compareHist(hist, new_hist, cv2.HISTCMP_CORREL)
            if score > best_score:
                best_score = score
                best_match = hist

        # Perform further actions based on the best match
        # Perform further actions based on the best match
        if best_match is not None:
            if objectId not in counted_object_id:
                # Person reappeared
                counted_object_id.append(objectId)
                print("Match found - Person reappeared: ID", objectId)
            else:
                # Person already counted, no need to notify
                pass
            # Your logic here based on the best match
            # Update enter/exit counts, draw lines/rectangles, etc.
            pass

        
        # Draw bounding box and ID on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = "ID: {}".format(objectId)
        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

    cv2.putText(frame, "Enters: " + str(enter_count), (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, "Exits: " + str(exit_count), (0, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, "# inside " + str(enter_count - exit_count), (0, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)



def selectEntrance(frame):
    # Create resizable window and set properties
    cv2.namedWindow('Select the entrance', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Select the entrance', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Select entrance ROI
    door = cv2.selectROI('Select the entrance', frame)

    # Destroy the entrance window
    cv2.destroyAllWindows()

    # Create resizable window and set properties
    



    return door

### print

#def expriemental

    
### main method


#classesFile = "coco.names";
#classes = None
#with open(classesFile, 'rt') as f:
#    classes = f.read().rstrip('\n').split('\n')
classes = 'person'

net = cv2.dnn.readNetFromDarknet("yolov4-tiny.cfg", "yolov4-tiny.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Path to the input video file
#Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
VIDEO_PATH = guiv1.give_source()
print(VIDEO_PATH)
#Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
#VIDEO_PATH = askopenfilename()
tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)
cap = cv2.VideoCapture(VIDEO_PATH)
if VIDEO_PATH != 0:
    outputFile = VIDEO_PATH[:-4]+'_out.avi'
elif VIDEO_PATH == 0:
    outputFile = 'CameraFootage'+'_out.avi'
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
    #isDetected = False
    #print("Getting frames.")


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
    used_boxes = postprocess(frame, outs)
    tracklines(used_boxes)
    while door is None:
        door = selectEntrance(frame)
    cv2.rectangle(frame, (door[0], door[1]), (door[0] + door[2], door[1] + door[3]), (0, 255, 255), 2)
    
    
    # Write the frame with the detection boxes
    vid_writer.write(frame.astype(np.uint8))
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Tracking", frame)
    
    # Write centroid_dict to a text file
    with open('centroid_dict.txt', 'w') as file:
        for key, value in centroid_dict.items():
            file.write(f"{key}: {value}\n")

    """ with open('ppl.txt', 'w') as file:
        for person_id, person_list in list_people.items():
            for person in person_list:
                file.write(str(person.person_id))
                file.write("\t")
                file.write(str(person.centroids))
                file.write("\t")
                file.write(str(person.histograms))  # Write histogram values
                file.write("\n")
            file.write("=-=-=-=-=-=-=-=-=-=")
            file.write("\n") """ 

