import albumentations as A
import cv2
import numpy as np
import os
import sys
import torch
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from ultralytics import YOLO

#
#YOLO + BOTSORT
#

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = YOLO("yolo11l.pt") # Set model

# Runs YOLO/Botsort
# Stream = True frees up memory, discards prior frames -- doesn't work for tracking, only detection
# Person is 0 class
# Bot-sort is included in ultralytics

vid_source='Clip.mp4' # Set video file here. Placeholder for testing
cap = cv2.VideoCapture(vid_source) # Video capture object for setting variables
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) # Get total frame count

results = model.track(source=vid_source, persist=True, save=False, classes=0, tracker="botsort.yaml") # Save for testing

b_boxes = {} # Bounding box dictionary for tracked objects

# Iterating thru each set of results. Each frame will have bounding box results for each tracked object.
for r in results:
    boxes = r.boxes

    for i, box in enumerate(boxes):
        xyxy = box.xyxy[0]  # Bounding box in [x_min, y_min, x_max, y_max]
        track_id = int(box.id[0]) # Tracking ID

        xyxy_clean = str(xyxy)
        xyxy_clean = xyxy_clean.replace("tensor(", "").replace(")", "").strip("[]")
        values = xyxy_clean.split(",")
        xyxy_clean = [int(float(value.strip())) for value in values]

        #Add ID to dict if not there and then append cleaned bounding box to its respective list
        if track_id not in b_boxes:
            b_boxes[track_id] = []
        b_boxes[track_id].append(xyxy_clean)

# Bounding box dictionary will have gaps for certain values since sometimes the tracked target is lost
# This will cause issues with running inference on the actions since it occurs frame-by-frame.
# Temporary solution that fills in bounding boxes with proximate values.
for track_id in b_boxes:
    while len(b_boxes[track_id]) < total_frames:
        b_boxes[track_id].append(b_boxes[track_id][-1])

#
#MVIT 
#

# See about adjusting these to improve performance? Currently default values.
clip_len = 16
show_video = False
resize_size = (224, 224)
crop_size = (224, 224)
weights = MViT_V2_S_Weights.KINETICS400_V1 # Options: DEFAULT, KINETICS400_V1

OUT_DIR = os.path.join('outputs')
os.makedirs(OUT_DIR, exist_ok=True)

# Define the transforms
transform = A.Compose([
    A.Resize(resize_size[1], resize_size[0]),
    A.Normalize(
        mean=[0.45, 0.45, 0.45],
        std=[0.225, 0.225, 0.225]
    )
])
print('Press q to quit')
model = mvit_v2_s(weights=weights).to(device).eval() 

# Load the labels file.
with open('kinetics_400_labels.csv', 'r') as f:
    class_names = f.readlines()
    f.close()

# Get the frame width and height.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))
save_name = 'TEST.mp4'.split('/')[-1].split('.')[0] # Test output file
show_video = True # True for testing purposes

# Define codec and create VideoWriter object.
out = cv2.VideoWriter(
    f"{OUT_DIR}/{save_name}.mp4", 
    cv2.VideoWriter_fourcc(*'mp4v'), 
    fps, 
    (frame_width, frame_height)
)

time_trackers = {} # Dictionary for time trackers for each object
VIDEO_FPS = int(cap.get(cv2.CAP_PROP_FPS)) # Constant for fps of the video file

#Running inference for each tracked object
for track_id in b_boxes:

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Resets capture position to beginning for new tracked object
    frame_count = 0 #  Frame count for b_box index
    clips = [] # List to append and store the individual frames
    time_tracker = {} # Dictionary to store labels and respective durations
    
    while(cap.isOpened()):

        ret, frame = cap.read()
        b_box = b_boxes[track_id][frame_count] # Current bounding box

        if ret == True:

            xmin, ymin, xmax, ymax = b_box
            cropped_frame = frame[ymin:ymax, xmin:xmax]

            image = cropped_frame.copy()
            cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            cropped_frame = transform(image=cropped_frame)['image']
            clips.append(cropped_frame)

            if len(clips) == clip_len:
                input_frames = np.array(clips)
                input_frames = np.expand_dims(input_frames, axis=0) # Add an extra dimension
                input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3)) # Transpose to get [1, 3, num_clips, height, width]

                # Convert the frames to tensor
                input_frames = torch.tensor(input_frames, dtype=torch.float32) 
                input_frames = input_frames.to(device)
                with torch.no_grad():
                    outputs = model(input_frames)
                
                _, preds = torch.max(outputs.data, 1) # Get the prediction index
                label = class_names[preds].strip() # Map predictions to the respective class names
                stripped_label = class_names[preds].strip().split(',')[-1].capitalize() #Reformat label for viewing
                frame_count += 1 # Increment frame count.

                time_tracker[stripped_label] = time_tracker.get(stripped_label, 0) + 1 / VIDEO_FPS #Add time & labels to dictionary

                cv2.putText(
                    image, 
                    f"{stripped_label}",
                    (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 255), 
                    1, 
                    lineType=cv2.LINE_AA
                )
                clips.pop(0)
                if show_video:
                    cv2.imshow('image', image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                out.write(image)
        else:
            break

    time_trackers[int(track_id)] = time_tracker
    
cap.release() # Release VideoCapture()
cv2.destroyAllWindows() # Close all frames and video windows

with open("Test-Report.txt", "w") as f:
    sys.stdout = f
    for track_id, time_tracker in time_trackers.items():
        print(f"\nTotal length of each action for Student {track_id}:")
        for label, total_time in time_tracker.items():
            print(f"{label}: {total_time:.2f}s")

sys.stdout = sys.__stdout__