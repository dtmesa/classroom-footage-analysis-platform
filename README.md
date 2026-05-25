# Classroom Footage Analysis Platform

<img width="2500" height="1666" alt="13_Spring25_Mesa" src="https://github.com/user-attachments/assets/0c0af4b6-327f-469c-aa92-1fdb1ade67a9" />

Cameras are becoming more common in classrooms, especially with the rising demand for hybrid & remote options. The footage captured offers valuable data for instructors & researchers, helping them analyze and improve the classroom–from the subject matter, to the activities, and to the various other facets of the learning experience.

Analyzing video footage can be a slow and time-consuming task. How might we streamline this process to more efficiently identify the types of activities, lengths of activities, and overall distribution of activities within classroom video footage?

A fine-tuned YOLO model is employed for object detection to identify students within the classroom. The tracking model BoT-SORT is used to maintain focus on each student as bounding boxes are collected. To eliminate duplicates generated during the tracking process, InsightFace, a facial recognition model is used. The video is cropped based on the bounding boxes from the tracking & object detection stage and processed using a narrowly trained action recognition model Video MViT. The duration of each detected action is recorded in frames and converted to time for each student.

## Features

- Fine-tuned YOLOv12 model for student detection
- BoT-SORT pedestrian tracking model to track detected students
- InsightFace facial recognition for duplicate tracking elimination
- Video MViT for action recognition and time accumulation

## Requirements

- The notebook for step 1 (Tracking/YOLO) requires Python 3.13.

- The notebook for steps 2 & 3 (InsightFace & MViT) requires Python 3.12.

- Virtual environments should be created to account for the difference in requirements.

## Running the files

- The labels file is required for the trained MVIT model.

- In Notebook 1, set the variable 'vid_source' to the location of your video footage

- In Notebook 1, ensure the file 'output_filename = "tracking_output.json"' is set to the same location as the file from Notebook 2 in the line 'with open("tracking_output.json", "r") as f:'

- In Notebook 2, you can rename the name of the output in the first line under the Report header: ("Test-Report.txt", "w")

If the virtual environments are correctly set up and the required packages are installed, simply run the first notebook and then the second. The first will generate output for the second and the second will generate a report for your viewing.

## Trained Models:

MVIT: https://drive.google.com/file/d/1cCxdr2G-RcjZtS1GcYhAtU2ca9awh3kY/view?usp=sharing

YOLO12l (Unused): https://drive.google.com/file/d/1sEK5vXEjg28-sCi3iKX3Oy8pJKWSznz1/view?usp=sharing

YOLO12m: https://drive.google.com/file/d/1CvDQQuzAYlOzVYN7-rBpOlJXiaBh3OMU/view?usp=sharing

## References

Fan, H., Xiong, B., Mangalam, K., Li, Y., Yan, Z., Malik, J., & Feichtenhofer, C. (2021). Multiscale Vision Transformers. arXiv. https://arxiv.org/abs/2104.11227

InsightFace. (n.d.). deepinsight/insightface. GitHub. https://github.com/deepinsight/insightface

Tian, Y., Ye, Q., & Doermann, D. (2025). YOLOv12: Attention-Centric Real-Time Object Detectors. arXiv preprint arXiv:2502.12524. https://arxiv.org/abs/2502.12524

Xu, Y., Li, B., Yuan, X., Yang, T., & Wang, G. (2022). HiEveTrack: Towards a benchmark for robust multi-object tracking in hierarchical human activities. arXiv. https://arxiv.org/abs/2206.14651

