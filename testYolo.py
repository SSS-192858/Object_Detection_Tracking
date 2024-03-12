import cv2
import numpy as np
from collections import defaultdict
from motmetrics import DistanceMetric, simotmetric
import time

class DeepSORT:
    def __init__(self):
        self.next_track_id = 0
        self.tracks = {}
        self.metric = DistanceMetric(kind='euclidean')

    def update(self, detections):
        matches, unmatched_tracks, unmatched_detections = self.associate_detections(detections)

        # Update existing tracks with new detections
        for track_id, detection_idx in matches:
            self.tracks[track_id]['boxes'].append(detections[detection_idx])
            self.tracks[track_id]['age'] += 1

        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = {'track_id': track_id, 'boxes': [detections[detection_idx]], 'age': 1}

        # Remove old tracks
        self.remove_old_tracks()

        # Return active tracks
        active_tracks = [track['boxes'][-1] for track_id, track in self.tracks.items() if track['age'] > 1]
        return active_tracks

    def associate_detections(self, detections):
        if not self.tracks:
            return [], [], list(range(len(detections)))

        track_boxes = [track['boxes'][-1] for track_id, track in self.tracks.items()]
        track_ids = list(self.tracks.keys())

        dist_matrix = self.metric.pairwise(np.array(track_boxes), np.array(detections))
        matches, unmatched_tracks, unmatched_detections = simotmetric(dist_matrix, 1)

        matches = [(track_ids[i], detection_idx) for i, detection_idx in matches]
        return matches, unmatched_tracks, unmatched_detections

    def remove_old_tracks(self, max_age=30):
        self.tracks = {track_id: track for track_id, track in self.tracks.items() if track['age'] <= max_age}


# Initialize DeepSORT tracker
deepsort = DeepSORT()

# Load YOLOv2 model
net = cv2.dnn.readNet("yolov2.weights", "yolov2.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Initialize video capture
cap = cv2.VideoCapture("input_video.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection with YOLOv2
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Parse detection results
    boxes = []
    confidences = []
    class_ids = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                w = int(detection[2] * frame_width)
                h = int(detection[3] * frame_height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform object tracking with DeepSORT
    detections = []
    for i, box in enumerate(boxes):
        detections.append([box[0], box[1], box[0] + box[2], box[1] + box[3]])
    active_tracks = deepsort.update(detections)

    # Draw bounding boxes and track IDs
    for track_box in active_tracks:
        x, y, x2, y2 = track_box
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

    # Write frame to output video
    out.write(frame)

    # Display frame
    cv2.imshow('Object Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer, and close windows
cap.release()
out.release()
cv2.destroyAllWindows()
