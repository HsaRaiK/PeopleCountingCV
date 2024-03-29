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
import guiv1

confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image
ttl_cntr = 0
selected = 0
centroid_dict = defaultdict(list)
object_id_list = []
enter = None
exit = None
enter_count = 0
exit_count = 0
counted_object_id = []
outed_object_id = []

def createTracker():
    tracker = cv2.legacy.TrackerKCF_create()
    return tracker

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

def detect(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []

    isDetected = False
    for out in outs:
        for detection in out:
            if isDetected == False:
               print("Detecting...")
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
        
        if classIds[i] == 0:
    	    ppl_cntr += 1
        
        #drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
        
        bbox = (left, top, left + width, top + height)

        used_boxes.append(bbox)
       
    cv2.putText(frame, str(ppl_cntr), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return used_boxes
        
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
                    start_pt = (centroid_dict[objectId][pt][0], centroid_dict[objectId][pt][1])
                    end_pt = (centroid_dict[objectId][pt + 1][0], centroid_dict[objectId][pt + 1][1])
                    cv2.line(frame, start_pt, end_pt, (0, 255, 0), 4)
                    
        if enter and exit is not None:      
            enter_count, exit_count = countPeople(frame, enter, exit, newbox, objectId)
            
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = "ID: {}".format(objectId)
        cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    cv2.putText(frame, str(enter_count), (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(frame, str(exit_count), (0, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
def selectEntrance(frame):
    enter = cv2.selectROI("Select the entrance", frame)
    cv2.destroyAllWindows()
    exit = cv2.selectROI("Select the exit", frame)
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
    
   #r1 inside r2
    
    if rx1 > enx1 and ry1 > eny1 and rx2 < enx2 and ry2 < eny2 and objectId not in counted_object_id:
        enter_count += 1
        counted_object_id.append(objectId)
        print(objectId, "arttı")
    if rx1 > ex1 and ry1 > ey1 and rx2 < ex2 and ry2 < ey2 and objectId not in outed_object_id:
        exit_count += 1
        outed_object_id.append(objectId)
        print(objectId, "azaldı")
        
    #if ry1 < eny2:
    #    count += 1
    #    print("c2")
    #if enx1 < rx1 < enx2 < rx2 and ry1 < eny2:
    #   count += 1
    #    print("c3")
    
    
    #if ey1 < ry2:
    #    count -= 1
    #    print("c4")
    
    #if rx1 < ex1 < rx2 < ex2 and ey1 < ry2:
    #    count -= 1
    #    print("c5")
    #if ex1 < rx1 < ex2 < rx2 and ey1 < ry2:
    #    count -= 1
    #    print("c6")
        

    return enter_count, exit_count

        


classesFile = "coco.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
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
    used_boxes = postprocess(frame, outs)
    tracklines(used_boxes)
    while exit is None:
        enter, exit = selectEntrance(frame)
    cv2.rectangle(frame, (enter[0], enter[1]), (enter[0] + enter[2], enter[1] + enter[3]), (0, 0, 255), 2)
    cv2.rectangle(frame, (exit[0], exit[1]), (exit[0] + exit[2], exit[1] + exit[3]), (0, 0, 255), 2)
    
    
    # Write the frame with the detection boxes
    vid_writer.write(frame.astype(np.uint8))
    cv2.imshow("Tracking", frame)
