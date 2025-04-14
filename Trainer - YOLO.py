from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    model = YOLO("yolo12x.pt")
    results = model.train(data=r"C:\Users\Dylan\Documents\School\Capstone\YOLO Training Data\data.yaml", epochs=100, imgsz=640,  batch=8)

if __name__ == "__main__":
    freeze_support()
    main()