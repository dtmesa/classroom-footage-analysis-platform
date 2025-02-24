import albumentations as A
import cv2
import numpy as np
import os
import torch
import time
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

cap = cv2.VideoCapture('Clip.mp4') # File to load

# See about adjusting these to improve performance
clip_len = 16
show_video = False
resize_size = (224, 224)
crop_size = (224, 224)
weights=MViT_V2_S_Weights.KINETICS400_V1 # Options: DEFAULT, KINETICS400_V1

OUT_DIR = os.path.join('outputs')
os.makedirs(OUT_DIR, exist_ok=True)

# Define the transforms
transform = A.Compose([
    A.Resize(resize_size[1], resize_size[0], always_apply=True),
    # A.CenterCrop(crop_size[1], crop_size[0], always_apply=True),
    A.Normalize(
        mean=[0.45, 0.45, 0.45],
        std=[0.225, 0.225, 0.225], 
        always_apply=True
    )
])
print('Press q to quit')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
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
show_video = True # True for testing purposes, false otherwise

# Define codec and create VideoWriter object.
out = cv2.VideoWriter(
    f"{OUT_DIR}/{save_name}.mp4", 
    cv2.VideoWriter_fourcc(*'mp4v'), 
    fps, 
    (frame_width, frame_height)
)

frame_count = 0 # To count 1 frames
total_fps = 0 # To get the final frames per second
clips = [] # List to append and store the individual frames
time_tracker = {} # Dictionary to store labels and respective durations
VIDEO_FPS = int(cap.get(cv2.CAP_PROP_FPS)) # Constant for the fps of the video file

# Read until end of video.
while(cap.isOpened()):
    # Capture each frame of the video.
    ret, frame = cap.read()
    if ret == True:
        # Get the start time.
        start_time = time.time()
        image = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(image=frame)['image']
        clips.append(frame)
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
            end_time = time.time() # Get the end time.
            fps = 1 / (end_time - start_time) # Get the fps.
            total_fps += fps # Add fps to total fps.
            frame_count += 1 # Increment frame count.

            time_tracker[stripped_label] = time_tracker.get(stripped_label, 0) + 1 / VIDEO_FPS #Add time & labels to dictionary

            cv2.putText(
                image, 
                f"Action: {stripped_label}",
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 0, 0), 
                2, 
                lineType=cv2.LINE_AA
            )
            cv2.putText(
                image, 
                f"Processing rate: {fps:.1f} fps", 
                (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 0, 0), 
                2, 
                lineType=cv2.LINE_AA
            )
            clips.pop(0)
            if show_video:
                cv2.imshow('image', image)
                # Press `q` to exit.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            out.write(image)
    else:
        break

cap.release() # Release VideoCapture()
cv2.destroyAllWindows() # Close all frames and video windows

print("\nTotal length of each action:")
for label, total_time in time_tracker.items():
    print(f"{label}: {total_time:.2f}s")