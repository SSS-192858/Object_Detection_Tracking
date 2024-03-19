import torch
import torchvision
import cv2
import numpy as np
from sort import sort
import time


start_time = time.time()
# Device configuration
device = torch.device('cpu')

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
model.eval()

# Initialize SORT tracker
object_tracker = sort.Sort(max_age=2800, min_hits=3)

# Constants
car_id = 2
confidence_threshold = 0.6

# Video I/O
cap = cv2.VideoCapture('./Videos/v1.avi')
width = int(cap.get(3))
height = int(cap.get(4))
out = cv2.VideoWriter('YoloV5_Sort.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))

# Set to store unique IDs
s = set()

# Main loop
while cap.isOpened():
    # Read the next frame
    ret, frame = cap.read()
    
    # Check if end of file
    if not ret:
        break
    
    # Run YOLOv5 prediction
    with torch.no_grad():
        pred = model(frame)

    # Extract detections for cars
    detections = []
    for index, detection in enumerate(pred.xyxy[0]):
        class_id = int(detection[5])
        confidence = detection[4].cpu().detach().numpy().item()

        if confidence > confidence_threshold and class_id == car_id:
            x_min, y_min, x_max, y_max = detection[:4].cpu().detach().numpy().astype('int')
            detections.append([x_min, y_min, x_max, y_max])
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)

    # Convert detections to numpy array
    detections = np.array(detections)

    # Update tracks with SORT
    if len(detections) == 0:
        continue

    tracks = object_tracker.update(detections)
    for id, track in enumerate(tracks):
        s.add(int(track[4]))

    # Display count
    cv2.putText(frame, f'Count: {len(s)}', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    # Write frame
    out.write(frame)

# Release video writer and capture object
out.release()
cap.release()
end_time = time.time()
print("The total time taken is : ",end_time-start_time," seconds.")