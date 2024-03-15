import cv2
import numpy as np
from sort.sort import Sort
import time
class YOLOv2Tracker:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        self.net = cv2.dnn.readNet(r"C:\Users\Srinivasan M\OneDrive\Desktop\Semester_6\VR\Object_Detection_Tracking\yolov2.weights",r"C:\Users\Srinivasan M\OneDrive\Desktop\Semester_6\VR\Object_Detection_Tracking\yolo-voc.cfg")
        self.classes = []
        with open("./coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.tracker = Sort()

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
                    boxes.append([x, y, x+w, y+h])  # Modified to fit SORT input format
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

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Object detection with YOLOv2
            boxes, confidences, class_ids = self.detect_objects(frame)

            # Combine boxes and confidences into detections for SORT
            detections = np.array([[x1, y1, x2, y2, conf] for (x1, y1, x2, y2), conf in zip(boxes, confidences)])

            # Update SORT tracker
            tracked_objects = self.tracker.update(detections)

            # Draw bounding boxes and track IDs
            for obj in tracked_objects:
                bbox = obj[:4].astype(int)
                track_id = int(obj[4])
                class_id = track_id % len(self.classes)
                color = (255, 0, 0)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(frame, str(track_id), (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

            print(f"Frames Per Second : {fps}")
            out.write(frame)


# Usage:
start_time = time.time()
tracker = YOLOv2Tracker("./Videos/V1.avi", "output_yolo_sort.avi")
tracker.track_objects()
end_time = time.time()
print("The time taken to process the video is : ", end_time - start_time, " seconds")
