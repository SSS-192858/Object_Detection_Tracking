import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load Faster R-CNN
device = torch.device('cpu')
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
model.to(device)
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
conf_threshold = 0.5

cap = cv2.VideoCapture("./Videos/V1.avi")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

transform = T.Compose([
    T.ToTensor(),
])
# frame_count = 0

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # frame_count += 1
    # if frame_count % 2 != 0:
        # continue

    frame_tensor = transform(frame).to(device)
    with torch.no_grad():
        outputs = model([frame_tensor])

    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        if score < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        color = (255, 0, 0)  # Red color for the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, classes[label], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    frame = cv2.resize(frame, (800, 600))   
    # Display frame
    out.write(frame)
    # cv2.imshow('Object Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
