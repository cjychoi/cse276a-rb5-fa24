# inspiration from:
# https://stackoverflow.com/questions/76069484/obtaining-detected-object-names-using-yolov8

# hamzakhan2018 on github

import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # YOLOv8 nano model

# Initialize the camera feed (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Input object name to detect
target_object = input("Enter the name of the object to detect: ")

# Focal length (hypothetical) and known width (could be pre-determined or based on the object)
KNOWN_WIDTH = 8.0  # Width of the object in cm (adjust based on the object)
FOCAL_LENGTH = 615  # Focal length of the camera in pixels (this is an example)

# Function to estimate distance based on bounding box size
def estimate_distance(bbox_width):
    if bbox_width > 0:
        distance = (KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width
        return distance
    else:
        return None

def temp():
    quit()

while True:
    # Capture frame-by-frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Detect objects using YOLOv8
    results = model(frame)

    for result in results:
        detection_count = result.boxes.shape[0]
        print('detection_count: ', detection_count)
        for i in range(detection_count):
            cls = int(result.boxes.cls[i].item())
            name = result.names[cls]
            print('name: ', name)
            if name == target_object:
                print('object found: ', target_object)

                bounding_box = result.boxes.xyxy[i].cpu().numpy()
                x = int(bounding_box[0])
                y = int(bounding_box[1])
                width = int(bounding_box[2] - x)
                height = int(bounding_box[3] - y)

                # Estimate the distance to the object
                distance = estimate_distance(width)
                print(distance)

                temp()

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()                                                                               