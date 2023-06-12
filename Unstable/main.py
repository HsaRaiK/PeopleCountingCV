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
import class_person as pepd
from datetime import datetime


confThreshold = 0.6  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image
ttl_cntr = 0
selected = 0
centroid_dict = defaultdict(list)
object_id_list = []
door = None
output_file = 'output.txt' 

enter_count = 0
exit_count = 0
counted_object_id = []
outed_object_id = []
rgb_values_list = []

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
    limitX = int(frameWidth)
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

        # Draw orange lines to separate the segments
        for j in range(1, 3):
            part_top = top + (j * part_height)
            cv2.line(frame, (left, part_top), (left + width, part_top), (0, 165, 255), 2)

        # Create a new person instance and store RGB values
        person = pepd.Person(bbox)
        for j in range(1, 4):
            part_top = top + ((j - 1) * part_height)
            part_bottom = part_top + part_height
            part_bbox = (left, part_top, left + width, part_bottom)
            part_frame = frame[part_top:part_bottom, left:left + width]
            avg_rgb = np.mean(part_frame, axis=(0, 1))

            if j == 1:
                person.rgb_value1 = avg_rgb
            elif j == 2:
                person.rgb_value2 = avg_rgb
            elif j == 3:
                person.rgb_value3 = avg_rgb

            print_box_info(box_id_counter, avg_rgb, output_file)
            # ...

        # Add the person object to the list
        #listofPeople.append(person)
        # ...
    return used_boxes

def print_box_info(box_id, avg_rgb, output_file):
    print("Box ID: {}".format(box_id))
    print("RGB values for each part:")
    rgb_values_list.append(avg_rgb)
    for j in range(1, 4):
        print("Part {}: {}".format(j, avg_rgb))
    with open(output_file, 'a') as file:
        file.write("Box ID: {}\n".format(box_id))
        file.write("RGB values for each part:\n")
        for j in range(1, 4):
            file.write("Part {}: {}\n".format(j, avg_rgb))


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
            cv2.line(frame, start_pt, end_pt, (255, 0, 0), 1)
            
        else:
            l = len(centroid_dict[objectId])
            for pt in range(len(centroid_dict[objectId])):
                if not pt + 1 == l:
                    start_pt = (centroid_dict[objectId][pt][0], centroid_dict[objectId][pt][1])
                    end_pt = (centroid_dict[objectId][pt + 1][0], centroid_dict[objectId][pt + 1][1])
                    cv2.line(frame, start_pt, end_pt, (0, 255, 0), 1)
        #object count starts from here:            
        if door is not None:      
            #enter_count, exit_count = countPeople(frame, door, newbox, objectId)
            enx1, eny1, enx2, eny2 = door
            mideny = int(eny1 + eny2 / 2)
            midenx = int(enx2 + enx1)
            #line start x: enx1
            #line start y: mideny
            #line end x= midenx
            #line end y: mideny 
            
            cv2.line(frame, (enx1, mideny), (midenx, mideny), (255, 0, 0), 2)
            
            l = len(centroid_dict[objectId])
            
            for pt in range(len(centroid_dict[objectId])):
               if not pt + 1 == l:
                    if enx1 < start_pt[0] and start_pt[0] < midenx:
                        if mideny > start_pt[1] and mideny <= end_pt[1] and objectId not in outed_object_id:
                            print("çık")
                            exit_count += 1
                            outed_object_id.append(objectId)
                            print(objectId)
                        if mideny < start_pt[1] and mideny >= end_pt[1] and objectId not in counted_object_id:
                	        print("gir")
                	        enter_count += 1
                	        counted_object_id.append(objectId)
                	        print(objectId)
                    else:
                        continue
            
            
            
                    
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = "ID: {}".format(objectId)
        cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        
        
        
     
    
    
        
        
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

def write_start_end(file_path):
    with open(file_path, 'a') as file:
        file.write("start: {}\n".format(datetime.now()))       


#classesFile = "coco.names";
#classes = None
#with open(classesFile, 'rt') as f:
#    classes = f.read().rstrip('\n').split('\n')
classes = 'person'

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

write_start_end(output_file)
# Read first frame
success, frame = cap.read()
# quit if unable to read the video file
if not success:
  print('Failed to read video')
  sys.exit(1)

frame_check = int(guiv1.give_frame_counter())
frame_counter = 0
while cv2.waitKey(1) < 0:

    # get frame from the video
    hasFrame, frame = cap.read()
    #isDetected = False
    #print("Getting frames.")
    frame_counter += 1

    
    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv2.waitKey(3000)
        # Release device
        cap.release()
        break

    if frame_counter % frame_check == 0:

        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

            # Sets the input to the network
        net.setInput(blob)

            # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

            # Remove the bounding boxes with low confidence
        used_boxes = postprocess(frame, outs)
            
        while door is None:
                door = selectEntrance(frame)
        cv2.rectangle(frame, (door[0], door[1]), (door[0] + door[2], door[1] + door[3]), (0, 255, 255), 2)
        #if frame_counter % frame_check == 0:
        tracklines(used_boxes)
        
    # Write the frame with the detection boxes
    vid_writer.write(frame.astype(np.uint8))
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Tracking", frame)
