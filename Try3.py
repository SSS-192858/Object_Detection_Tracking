import cv2
import numpy as np
from deep_sort.deep_sort import nn_matching
import time
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker

class YOLOv2Tracker:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        self.net = cv2.dnn.readNet("yolov2.weights", "yolo-voc.cfg")
        self.classes = []
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_objects(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

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
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        return boxes, confidences, class_ids

    def track_objects(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 20
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))

        # DeepSORT setup
        max_cosine_distance = 0.3
        nn_budget = None
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Object detection with YOLOv2
            boxes, confidences, class_ids = self.detect_objects(frame)

            # Perform object tracking with DeepSORT
            detections = [Detection(bbox, score, np.array([])) for bbox, score in zip(boxes, confidences)]
            tracker.predict()
            tracker.update(detections)

            # Draw bounding boxes and track IDs
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                bbox = bbox.astype(int)  # Convert bounding box coordinates to integers
                class_id = track.track_id % len(self.classes)  # Use track ID as class ID for visualization
                color = (255, 0, 0)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(frame, str(track.track_id), (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)
            # frame = cv2.resize(frame, (800, 600))
            # start_time = time.time()  # We would like to measure the FPS.
            # # frame = self.draw_boxes(frame)  # Plot the boxes directly
            # end_time = time.time()
            # fps = 1 / np.round(end_time - start_time, 3)  # Measure the FPS.
            print(f"Frames Per Second : {fps}")
            out.write(frame)

# Usage:
tracker = YOLOv2Tracker("./Videos/V1.avi", "output_yolo.avi")
tracker.track_objects()
