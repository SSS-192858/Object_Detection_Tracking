import cv2
import time
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

stream = cv2.VideoCapture('/Users/vikaskaly/Downloads/IIITB-Courses/SEM-6/VR/Assignment3/Traffic Video short.mp4') # 0 means read from local camera

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
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

class FasterRCNNDetector:
    
    def __init__(self, model, stream):    
        self.model = model
        self.stream = stream
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.transform = T.Compose([
            T.ToTensor(),
        ])

    def plot_boxes(self, frame):
        frame_np = frame.copy()  # Ensure frame is a numpy array
        if not isinstance(frame_np, np.ndarray):
            raise ValueError("Frame is not a valid numpy array")

        frame_tensor = self.transform(frame_np).to(self.device)
        with torch.no_grad():
            outputs = self.model([frame_tensor])

        boxes = outputs[0]['boxes'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()
        labels = outputs[0]['labels'].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            # If score is less than 0.5 we avoid making a prediction.
            if score < 0.5:
                continue
            x1, y1, x2, y2 = map(int, box)
            bgr = (0, 255, 0)  # color of the box
            label_font = cv2.FONT_HERSHEY_SIMPLEX  # Font for the label.
            cv2.rectangle(frame_np, (x1, y1), (x2, y2), bgr, 2)  # Plot the boxes
            cv2.putText(frame_np, classes[label], (x1, y1), label_font, 0.9, bgr, 2)  # Put a label over box.
        return frame_np


    def __call__(self, out_file):
        assert self.stream.isOpened()

        x_shape = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = cv2.VideoWriter_fourcc(*"MJPG")  # Using MJPEG codec
        out = cv2.VideoWriter(out_file, four_cc, 20, (x_shape, y_shape))
        rect, frame = self.stream.read()  # Read the first frame.
        while rect:
            start_time = time.time()  # We would like to measure the FPS.
            frame = self.plot_boxes(frame)  # Plot the boxes directly
            end_time = time.time()
            fps = 1 / np.round(end_time - start_time, 3)  # Measure the FPS.
            print(f"Frames Per Second : {fps}")
            out.write(frame)  # Write the frame onto the output.
            rect, frame = self.stream.read()  # Read next frame.

faster_rcnn_detector = FasterRCNNDetector(model, stream)
faster_rcnn_detector("output_RCNN.avi")  # Call the function to start the process.