import torch
import torchvision
import cv2
import time
from PIL import Image
from torchvision import transforms as T
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


start_time = time.time()
# Setting up device
device = torch.device('cpu')

# Constants
car_class_id = 2
confidence_val = 0.5
object_type = ['car']

# Open video file
cap = cv2.VideoCapture('./Videos/V1.avi')
width = int(cap.get(3))
height = int(cap.get(4))

# Video writer
out = cv2.VideoWriter('YoloV5_DeepSort.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))

# Initialize set for tracking unique IDs
s = set()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
model.eval()

# Initialize DeepSORT tracker
object_tracker = DeepSort(max_iou_distance=0.8, max_age=250, nms_max_overlap=0.05,
                           gating_only_position=True, n_init=2, max_cosine_distance=0.9)

# Main loop
while cap.isOpened():
    # Read frame
    ret, frame = cap.read()

    # Check if frame is read successfully
    if not ret:
        break

    # Perform inference with YOLOv5
    with torch.no_grad():
        pred = model(frame)

    # Extract detections for cars
    detections = []
    for detection in pred.xyxy[0]:
        class_id = int(detection[5])
        confidence = detection[4]
        if confidence > confidence_val and class_id == car_class_id:
            x_min, y_min, x_max, y_max = detection[:4].cpu().detach().numpy().astype('int')
            x, y, w, h = [x_min, y_min, int(x_max - x_min), int(y_max - y_min)]
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
            detections.append(([x, y, w, h], confidence, 'car'))

    # Update tracks with DeepSORT
    tracks = object_tracker.update_tracks(detections, frame=frame)
    
    # Draw tracks
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltrb()
        s.add(track_id)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        cv2.putText(frame, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Vehicle count
    # count = object_tracker.tracker._next_id - 1
    cv2.putText(frame, f'Vehicle Count: {len(s)}', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    # Write frame
    out.write(frame)

# Release video writer and capture object
out.release()
cap.release()
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
