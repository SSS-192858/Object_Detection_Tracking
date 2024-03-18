import torch 
import torchvision 
import cv2 
from PIL import Image 
from torchvision import transforms as T 
import numpy as np 
from deep_sort_realtime.deepsort_tracker import DeepSort 

device = torch.device('cpu')

car_class_id = 2 
confidence_val = 0.6
object_type = ['car']

cap = cv2.VideoCapture('./Videos/v1.avi')
width = int(cap.get(3))
height = int(cap.get(4))
out = cv2.VideoWriter('YoloV5.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))
s = set()
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
model.eval()
object_tracker = DeepSort(max_iou_distance=0.8,max_age=250,nms_max_overlap = 0.05, gating_only_position=True, n_init=2, 
max_cosine_distance=0.9)

count = 0
while cap.isOpened():
    # Read the next frame
    ret, frame = cap.read()
    # Check EOF 
    if not ret:
        break

    with torch.no_grad(): 
        pred = model(frame)
        
    #Creating Detection data for cars to pass into SORT tracks directly 
    detections = [] 
    for detection in pred.xyxy[0]:
        class_id = int(detection[5])
        confidence = detection[4]
        if confidence> confidence_val and class_id ==car_class_id:
            x_min,y_min,x_max,y_max = detection[:4].cpu().detach().numpy().astype('int')
            x,y,w,h = [x_min,y_min,int(x_max-x_min),int(y_max-y_min)]
            
            #Bounding Box and id  
            cv2.rectangle(frame,(int(x_min), int(y_min)),(int(x_max), int(y_max)),(0,0,255),2)
            
            #formatting in form which deepsort accepts
            detections.append(([x,y,w,h],confidence,'car'))
            
    #Tracks for DeepSORT and Annotate 
    tracks = object_tracker.update_tracks(detections, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltrb()
        s.add(track_id)
        # count+=1
        cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(0,0,255),2)
        cv2.putText(frame, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    #Count 
    count = object_tracker.tracker._next_id-1 
    cv2.putText(frame, f'Vehicle Count: {len(s)}', (20,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    print("Hello")
    out.write(frame)
out.release()