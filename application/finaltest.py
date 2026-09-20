from ultralytics import YOLO
import torch #pytorch specially for cuda support
import cv2 #open cv

model = YOLO(r"add path to your best.pt model ")
results = model.predict(
    source=r"add path to you .mp4 / png file",
    save=True,  
    conf=0.4,  #confidence check
    project=r"path where result should be saved",  # where to save
    name="pothole_output",  #name of subfolder for the saved result
                            # subfolder name
)

