import torch 
import torchvision 
import cv2 
from PIL import Image 
from torchvision import transforms as T 
import numpy as np 
import sys
# sys.path.insert(0,'./sort/')
from sort import sort


device = torch.device('cpu')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
model.eval()

object_tracker = sort.Sort(max_age=2800, min_hits=3)

s = set()
#Constants 
car_id = 2 
confidence_threshold = 0.6
object_type = ['car']
count = 0 

#Video I/O
cap = cv2.VideoCapture('./Videos/v1.avi')
width = int(cap.get(3))
height = int(cap.get(4))
out = cv2.VideoWriter('output3.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))

#storage 
bounding_boxes = np.array([])

#model parameters
model.eval()

#main loop 
while cap.isOpened():
    # Read the next frame
    ret, frame = cap.read()
    # Check EOF 
    if not ret:
        break
    #prediction YOLO 
    with torch.no_grad(): 
      pred = model(frame)
        
    #Creating Detection data for cars to pass into SORT tracks directly 
    detections = [] 
    # indices = []
    for index,detection in enumerate(pred.xyxy[0]):
        class_id = int(detection[5])
        confidence = detection[4].cpu().detach().numpy().item()

        if confidence> confidence_threshold and class_id ==car_id:
            x_min,y_min,x_max,y_max = detection[:4].cpu().detach().numpy().astype('int')
            
            #formatting in form which sort accepts
            detections.append([x_min,y_min,x_max,y_max])
            
            x,y,w,h = [x_min,y_min,int(x_max-x_min),int(y_max-y_min)]

            #Bounding Box and id  
            cv2.rectangle(frame,(int(x_min), int(y_min)),(int(x_max), int(y_max)),(0,0,255),2)
            
    detections = np.array(detections)        
            
    #Tracks for SORT 
    if(len(detections)==0): continue 
    
    tracks = object_tracker.update(detections)
    for id,track in enumerate(tracks):
        s.add(int(track[4]))
        bounding_boxes=np.append(bounding_boxes,track[4])
    
    # temp = np.unique(bounding_boxes)
    # count = temp.shape[0]

    #Count 
    cv2.putText(frame, f'Count: {len(s)}', (20,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    
    out.write(frame)
out.release()


print(count)