from ultralytics import YOLO

model = YOLO(r"add path to your best.pt model")  # your latest trained model

results = model.predict(
    source=0,        # 0 = default webcam
    show=True,        # opens a live window showing detections
    conf=0.60,
    device=0
)