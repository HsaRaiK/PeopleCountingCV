import cv2
import numpy as np
from tkinter import Tk     # pip install tk
from tkinter.filedialog import askopenfilename

import argparse

# Path to the input video file
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
VIDEO_PATH = askopenfilename()
print(VIDEO_PATH)

# Path to the YOLO model weights file
MODEL_WEIGHTS_PATH = 'yolov3.weights'

# Path to the YOLO model configuration file
MODEL_CONFIG_PATH = 'yolov3-1.cfg'

# Path to the label map file
LABEL_MAP_PATH = 'yolov3.txt'

# Load the label map
with open(LABEL_MAP_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load the YOLO model
model = cv2.dnn.readNetFromDarknet(MODEL_CONFIG_PATH, MODEL_WEIGHTS_PATH)

# Create a tracker
tracker = cv2.TrackerKCF_create()

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)

# Initialize the tracker with the first frame
success, frame = cap.read()
if not success:
    print('Failed to read the video file')
    exit()

bboxes = []
scores = []
classes = []

while True:
    # Read the next frame
    success, frame = cap.read()
    if not success:
        break

    # Prepare the input image
    #image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = frame
    image = cv2.resize(image, (320, 320))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    # Run the detection
    model.setInput(image)
    out = model.forward()

    # Extract the bounding boxes of the 'person' class
    for i, detect in enumerate(out[0, 0, :, :]):
        score = detect[5:]
        class_id = np.argmax(score)
        confidence = score[class_id]
        if confidence > 0.5 and labels[class_id] == 'person':
            x, y, w, h = int(detect[0] * frame.shape[1]), int(detect[1] * frame.shape[0]), int(detect[2] * frame.shape[1]), int(detect[3] * frame.shape[0])
            bboxes.append((x, y, w, h))
            scores.append(float(confidence))
            classes.append(class_id)

    # Draw the bounding boxes and labels
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        label = labels[classes[i]]
        color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x+w,y+h), color, 2)
        cv2.putText(image, label, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Update the tracker with the new bounding boxes
    if bboxes:
        success, bboxes = tracker.update(image, bboxes)
        if not success:
            tracker = cv2.TrackerKCF_create()
            success, bboxes = tracker.update(image, bboxes)

    # Show the image
    cv2.imshow('image', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy the windows
cap.release()
cv2.destroyAllWindows()