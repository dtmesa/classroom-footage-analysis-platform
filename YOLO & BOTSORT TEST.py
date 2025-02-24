import sys
import torch
from ultralytics import YOLO

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = YOLO("yolov8n.pt") # Testing with yolov8n, actual with yolo11m-seg.pt

# Runs YOLO
# Stream = True frees up memory, discards prior frames -- doesn't work for tracking, only detection
# Person is 0 class
# Bot-sort is included in ultralytics

results = model.track(source="Clip.mp4", persist=True, save=True, classes=0, tracker="botsort.yaml") # Save for testing

# #Testing
# print()
# print("RESULTS: \n")
# print(results[0])

# For each box, print the coordinates, confidence, and class
with open("output.txt", "w") as f:
    sys.stdout = f

    for r in results:
        boxes = r.boxes

        # Iterate through each box and extract details
        for i, box in enumerate(boxes):
            xyxy = box.xyxy[0]  # Bounding box in [x_min, y_min, x_max, y_max]
            track_id = box.id[0] if box.id is not None else "N/A"  # Tracking ID (if available)
            
            # Print out details for each tracked object
            print(f"Tracked Object {int(track_id)}: Coordinates: {xyxy}")

    sys.stdout = sys.__stdout__

print("Output saved to output.txt")