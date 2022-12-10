wget https://pjreddie.com/media/files/yolov3.weights
python yolo_opencv.py --image dogandhum.jpg --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt
