#https://thepythoncode.com/article/real-time-object-tracking-with-yolov8-opencv

#use run 4
from ultralytics import YOLO
import cv2


# Load a model
#model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('runs/detect/train4/weights/best.pt')  # load a custom model

#load picture
path = 'walking_dog.jpeg'
# Predict with the model
results = model(path)  # predict on an image

i=0
while i < len(results):
    print(results[i])
    i += 1

print(type(results))
