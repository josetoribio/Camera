#https://thepythoncode.com/article/real-time-object-tracking-with-yolov8-opencv

#use run 4
from ultralytics import YOLO
import cv2


# define some constants
CONFIDENCE_THRESHOLD = 0.8
RED = (0, 0, 255)

# Load a model
#model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('runs/detect/train4/weights/best.pt')  # load a custom model

#load picture
path = 'bus.jpg'
# Predict with the model
detections = model(path)  # predict on an imageSSSS
#image reader for cv2
frame = cv2.imread(path)
#this gets the amount of detections
print(len(detections[0]))
print("type: ")
#this gets the class: ultralytics.engine.results.Results
print(type(detections[0][0]))

while True:
    for data in detections[0].boxes.data.tolist():
            # extract the confidence (i.e., probability) associated with the detection
            confidence = data[4]

            # filter out weak detections by ensuring the 
            # confidence is greater than the minimum confidence
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            # if the confidence is greater than the minimum confidence,
            # draw the bounding box on the frame
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), RED, 2)
            print(data[6])
            cv2.putText(frame, 'Fedex', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break


