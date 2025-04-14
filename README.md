# Classroom Footage Analysis Platform 🎥

Cameras are becoming more common in classrooms, especially with the rising demand for hybrid & remote options. The footage captured offers valuable data for instructors & researchers, helping them analyze and improve the classroom–from the subject matter, to the activities, and to the various other facets of the learning experience.

Analyzing video footage can be a slow and time-consuming task. How might we streamline this process to more efficiently identify the types of activities, lengths of activities, and overall distribution of activities within classroom video footage?

A fine-tuned YOLO model is employed for object detection to identify students within the classroom. The tracking model BoT-SORT is used to maintain focus on each student as bounding boxes are collected. To eliminate duplicates generated during the tracking process, InsightFace, a facial recognition model is used. The video is cropped based on the bounding boxes from the tracking & object detection stage and processed using a narrowly trained action recognition model Video MViT. The duration of each detected action is recorded in frames and converted to time for each student.

## Features

- Fine-tuned YOLOv12 model for student detection
- BoT-SORT pedestrian tracking model to track detected students
- InsightFace facial recognition for duplicate tracking elimination
- Video MViT for action recognition and time accumulation

# Requirements

The notebook for step 1 (Tracking/YOLO) requires Python 3.13.
The notebook for steps 2 & 3 (InsightFace & MViT) requires Python 3.12.
Virtual environments should be created to account for the difference in requirements.

# Running the files

In notebook 1,

# Trained Models:

MVIT: https://drive.google.com/file/d/1cCxdr2G-RcjZtS1GcYhAtU2ca9awh3kY/view?usp=sharing
YOLO12l: https://drive.google.com/file/d/1sEK5vXEjg28-sCi3iKX3Oy8pJKWSznz1/view?usp=sharing
YOLO12m: https://drive.google.com/file/d/1CvDQQuzAYlOzVYN7-rBpOlJXiaBh3OMU/view?usp=sharing
