import cv2
import numpy as np
import torch
import time
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load Faster R-CNN
device = torch.device('cpu')
rcnn= fasterrcnn_resnet50_fpn(pretrained=True)
rcnn.eval()
streamer = cv2.VideoCapture("./Videos/V1.avi")
rcnn.to(device)
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

# Threshold for confidence level
class FasterRCNN:

    def __init__(self,rcnn,stream,threshold=0.5):
        self.model = rcnn
        self.stream = stream
        self.device =  'cpu'
        self.model.to(self.device)
        self.threshold = threshold
        self.transform = T.Compose([
            T.ToTensor(),
        ])


    
    def draw_boxes(self,frame):
        frame1 = frame.copy()
        if isinstance(frame1,np.ndarray) == False:
            print("Invalid frame")
            exit(1)

        frame_tsor = self.transform(frame1).to(self.device)
        with torch.no_grad():
            output = self.model([frame_tsor])

        boxes = output[0]['boxes'].cpu().numpy()
        scores = output[0]['scores'].cpu().numpy()
        labels = output[0]['labels'].cpu().numpy()


        # Loop over the detected objects and annotate them on the frame
        for detected_box, detected_score, detected_label in zip(boxes, scores, labels):
            # Skip objects with low confidence
            if detected_score < self.threshold:
                continue

            # Extract coordinates of the bounding box
            x1, y1, x2, y2 = map(int, detected_box)
            
            # Define color and font for the label
            box_color = (0, 255, 0)  # Green color for the bounding box
            label_font = cv2.FONT_HERSHEY_SIMPLEX

            # Draw a rectangle around the detected object
            cv2.rectangle(frame1, (x1, y1), (x2, y2), box_color, thickness=2)
            
            # Add label to the detected object
            label_text = classes[detected_label]
            cv2.putText(frame1, label_text, (x1, y1), label_font, fontScale=0.9, color=box_color, thickness=2)
        # Return the frame with annotations
        return frame1
    
    def __call__(self, out_file):

        x_shape = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # Using MJPEG codec
        output_writer = cv2.VideoWriter(out_file, fourcc, 20, (x_shape, y_shape))
        rectangle, frame = self.stream.read()  # Read the first frame.
        while rectangle:
            start_time = time.time()  # We would like to measure the FPS.
            frame = self.draw_boxes(frame)  # Plot the boxes directly
            end_time = time.time()
            fps = 1 / np.round(end_time - start_time, 3)  # Measure the FPS.
            print(f"Frames Per Second : {fps}")
            output_writer.write(frame)  # Write the frame onto the output.
            rectangle, frame = self.stream.read()  # Read next f

faster_rcnn = FasterRCNN(rcnn, streamer,0.5)
faster_rcnn("output_RCNN.avi")  


