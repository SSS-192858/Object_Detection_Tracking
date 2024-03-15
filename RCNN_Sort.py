import cv2
import numpy as np
from sort.sort import Sort
import torch
import time
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class FasterRCNNSortTracker:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        
        # Load the pre-trained Faster R-CNN model from torchvision
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        
        # Move the model to CPU
        self.device = torch.device("cpu")
        self.model.to(self.device)
        
        # Define the transformations to be applied to the input frame
        self.transform = transforms.Compose([transforms.ToTensor()])
        
        # Initialize the SORT tracker
        self.tracker = Sort()

    def detect_objects(self, frame):
        # Apply transformations to the input frame
        input_tensor = self.transform(frame).to(self.device)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        # Extract predicted bounding boxes, labels, and scores
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        
        # Filter out detections with scores less than 0.5
        threshold = 0.5
        boxes = boxes[scores > threshold]
        
        return boxes

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
            boxes = self.detect_objects(frame)

            # Combine boxes into detections for SORT
            detections = np.array([[x1, y1, x2, y2, 1] for (x1, y1, x2, y2) in boxes])  # Assign a confidence of 1 for all detections

            # Update SORT tracker
            tracked_objects = self.tracker.update(detections)

            # Draw bounding boxes and track IDs
            for obj in tracked_objects:
                bbox = obj[:4].astype(int)
                track_id = int(obj[4])
                color = (255, 0, 0)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(frame, str(track_id), (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

            print(f"Frames Per Second : {fps}")
            out.write(frame)

# Usage:
start_time = time.time()
tracker = FasterRCNNSortTracker("./Videos/V1.avi", "output_faster_rcnn_sort.avi")
tracker.track_objects()
end_time = time.time()
print("The total time taken is : ",end_time-start_time," seconds.")

