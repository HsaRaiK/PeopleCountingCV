#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################


import cv2
import argparse
import numpy as np
import random as r
from pathlib import Path

arr_ppl = []

def idchecker(id, arr_ppl):
    for x in arr_ppl:
        if x.number != id:
            return True
        else:
            return False

class personId:
    def __init__(self,number):
        #id = r.randint(1000,9999)+ppl_cntr
        #condtionId = idchecker(id, arr_ppl)
        #if (condtionId):
        #    self.number = id
        #else:        
        #    id = r.randint(1000,9999)+ppl_cntr
        #    self.number = id
        self.number = number


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, lbl_id):

    if (class_id == 0):
        label = str(lbl_id)
        color = (0,252,0)
    else:
        label = str(classes[class_id])
        color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


image = cv2.imread(args.image)

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(args.weights, args.config)

blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

net.setInput(blob)

outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4


for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])



indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
ppl_cntr = 0;

for i in indices:
    try:
        box = boxes[i]
    except:
        i = i[0]
        box = boxes[i]
    
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    if class_ids[i] == 0:
        ppl_cntr += 1
        person_found = personId(ppl_cntr)
        arr_ppl.append(person_found)

        
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), person_found.number)
    else:
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), 0)

cv2.putText(image, str(ppl_cntr), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 2)


cv2.imshow("object detection", image)
cv2.waitKey()

print(len(arr_ppl))
cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()
