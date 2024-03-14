import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker

class FasterRCNNTracker:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.device = torch.device('cpu')
        self.model.to(self.device)
        self.transform = T.Compose([
            T.ToTensor(),
        ])
        self.classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                        "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
                        "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
                        "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                        "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                        "hair drier", "toothbrush"]
        self.tracker = self.initialize_tracker()

    def initialize_tracker(self):
        # DeepSORT setup
        max_cosine_distance = 0.3
        nn_budget = None
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)
        return tracker

    def detect_objects(self, frame):
        frame_tensor = self.transform(frame).to(self.device)
        with torch.no_grad():
            outputs = self.model([frame_tensor])

        boxes = outputs[0]['boxes'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()
        class_ids = outputs[0]['labels'].cpu().numpy()

        return boxes, scores, class_ids

    def track_objects(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 20 
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Object detection with Faster R-CNN
            boxes, scores, class_ids = self.detect_objects(frame)

            # Perform object tracking with DeepSORT
            detections = [Detection(bbox, score, class_id) for bbox, score, class_id in zip(boxes, scores, class_ids)]
            self.tracker.predict()
            self.tracker.update(detections)

            # Draw bounding boxes and track IDs
            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                bbox = bbox.astype(int)  # Convert bounding box coordinates to integers
                class_id = track.track_id % len(self.classes)  # Use track ID as class ID for visualization
                color = (255, 0, 0)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(frame, str(track.track_id), (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

            out.write(frame)

# Usage:
tracker = FasterRCNNTracker("./Videos/V1.avi", "output_rcnn.avi")
tracker.track_objects()
