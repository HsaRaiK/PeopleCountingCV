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
from skimage.feature import hog
	
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

def compute_histogram(frame, boxes, histograms):
       for i, box in enumerate(boxes):
           left, top, width, height = box
           roi = frame[top:top+height, left:left+width]
           hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
           mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
           hist = cv2.calcHist([hsv_roi], [0, 1], mask, [180, 256], [0, 180, 0, 256])
           cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
           histograms[i] = hist
       return histograms
       
def compare_histograms(hist1, hist2):
	return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)     

def update(histograms, person_id, img, boxes):
	histograms = compute_histogram(img, boxes, histograms)
	histograms[person_id] = hist
	return histograms
	
def find_matching_id(img, boxes, histograms, threshold=0.5):
	new_hist = compute_histogram(img, boxes, histograms)
	print(new_hist)
	for person_id, stored_hist in histograms.items():
		print(stored_hist)
		if compare_histograms(new_hist, stored_hist) < threshold:
			return person_id
	return None	



def detect(frame, outs, confThreshold):
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
				if classId == 0:
					classIds.append(classId)
					confidences.append(float(confidence))
					boxes.append([left, top, width, height])
	return boxes, confidences, classIds	
	
def postprocess(frame, outs, confThreshold, nmsThreshold):
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
	boxes, confidences, classIds = detect(frame, outs, 0.6)
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

def selectEntrance(frame):
	cv2.namedWindow('Select the entrance', cv2.WINDOW_NORMAL)
	cv2.setWindowProperty('Select the entrance', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Select entrance ROI
	door = cv2.selectROI('Select the entrance', frame)

    # Destroy the entrance window
	cv2.destroyAllWindows()
	return door

def tracklines(frame, used_boxes, object_id_list, centroid_dict, enter_count, door, exit_count, counted_object_id, 
outed_object_id, tracker, histograms):
	objects = tracker.update(used_boxes)
	
	histograms = {}
	new_hist = {}
	
	for (objectId, newbox) in objects.items():
		x1, y1, x2, y2 = newbox #x, y, w, h
		x1 = int(x1)
		y1 = int(y1)
		x2 = int(x2)
		y2 = int(y2)

		cX = int((x1 + x2) / 2.0)
		cY = int((y1 + y2) / 2.0)
		cv2.circle(frame, (cX, cY), 4, (0, 255, 0), -1)

		centroid_dict[objectId].append((cX, cY))
		

		
		person_id = find_matching_id(frame, used_boxes, histograms)
		
		
		if person_id is None:
				person_id = f"Person {len(histograms) + 1}"
				hist, histograms = update(histograms, person_id, frame, used_boxes)
		
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
        #object count starts from here:            
		if door is not None:      
			enx1, eny1, enx2, eny2 = door
			mideny = int(eny1 + eny2 / 2)
			midenx = int(enx2 + enx1)
			
			cv2.line(frame, (enx1, mideny), (midenx, mideny), (255, 0, 0), 2)
			
			l = len(centroid_dict[objectId])
            
			for pt in range(len(centroid_dict[objectId])):
				if not pt + 1 == l:
					if enx1 < start_pt[0] and start_pt[0] < midenx:
						if mideny > start_pt[1] and mideny <= end_pt[1] and objectId not in outed_object_id:
							print("çık")
							exit_count += 1
							outed_object_id.append(objectId)
                            
						if mideny < start_pt[1] and mideny >= end_pt[1] and objectId not in counted_object_id:
							print("gir")
							enter_count += 1
							counted_object_id.append(objectId)
							print(objectId)
					else:
						continue

		
		
		cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
	
	for person_id, stored_hist in histograms.items():
		text = "ID: {}".format(person_id)
		cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
		

	cv2.putText(frame, "Enters: " + str(enter_count), (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
	cv2.putText(frame, "Exits: " + str(exit_count), (0, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
	cv2.putText(frame, "# inside " + str(enter_count - exit_count), (0, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
	
	return enter_count, exit_count, histograms

def main():
	ttl_cntr = 0
	selected = 0
	centroid_dict = defaultdict(list)
	object_id_list = []
	door = None
	confThreshold = 0.6  #Confidence thresholddnn
	nmsThreshold = 0.4   #Non-maximum suppression threshold
	inpWidth = 416       #Width of network's input image
	inpHeight = 416 
	enter_count = 0
	exit_count = 0
	counted_object_id = []
	outed_object_id = []
	histograms = {}
	net = cv2.dnn.readNetFromDarknet("yolov4-tiny.cfg", "yolov4-tiny.weights")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
	
	VIDEO_PATH = guiv1.give_source()

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

	if not success:
  		print('Failed to read video')
  		sys.exit(1)

	while cv2.waitKey(1) < 0:
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
		used_boxes = postprocess(frame, outs, confThreshold, nmsThreshold)
		
		enter_count, exit_count, histograms = tracklines(frame, used_boxes, object_id_list, centroid_dict, enter_count, door, exit_count, counted_object_id, 
outed_object_id, tracker, histograms)
		
		while door is None:
			door = selectEntrance(frame)
		cv2.rectangle(frame, (door[0], door[1]), (door[0] + door[2], door[1] + door[3]), (0, 0, 255), 2)
		
		vid_writer.write(frame.astype(np.uint8))
		cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
		cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
		cv2.imshow("Tracking", frame)

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
