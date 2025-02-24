import albumentations as A
import cv2
import numpy as np
import os
import torch
import time
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from ultralytics import YOLO

#
#YOLO + BOTSORT
#

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = YOLO("yolo11n.pt") # Set model. Set to 11n for testing. More resource intensive models available.

# Runs YOLO/Botsort
# Stream = True frees up memory, discards prior frames -- doesn't work for tracking, only detection
# Person is 0 class
# Bot-sort is included in ultralytics

vid_source='Clip.mp4' # Set video file here. Placeholder for testing

results = model.track(source=vid_source, persist=True, save=False, classes=0, tracker="botsort.yaml") # Save for testing

#Bounding box dictionary for tracked objects
b_boxes = {}

# Iterating thru each set of results. Each frame will have bounding box results for each tracked object.
for r in results:
    boxes = r.boxes

    for i, box in enumerate(boxes):
        xyxy = box.xyxy[0]  # Bounding box in [x_min, y_min, x_max, y_max]
        track_id = box.id[0] if box.id is not None else "N/A"  # Tracking ID

        xyxy_clean = str(xyxy)
        xyxy_clean = xyxy_clean.replace("tensor(", "").replace(")", "").strip("[]")
        values = xyxy_clean.split(",")
        xyxy_clean = [int(float(value.strip())) for value in values]

        #Add ID to dict if not there and then append cleaned bounding box to its respective list
        if track_id not in b_boxes:
            b_boxes[track_id] = []
        b_boxes[track_id].append(xyxy_clean)