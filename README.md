# Pothole Detection Using YOLO

Detects potholes in road video footage using a custom-trained YOLO11 model, drawing bounding boxes around each detection.

## Hardware
Lenovo LOQ — Intel i5 (13th Gen), NVIDIA RTX 4050 (CUDA)

## Tech Stack
- Python 3.11
- Ultralytics YOLO11 (nano) — detection model
- PyTorch (CUDA) — deep learning engine
- OpenCV — video frame extraction/writing (auto-installed with ultralytics)
- Roboflow — pre-labeled dataset source

## Dataset
Pre-labeled pothole dataset from Roboflow Universe, split into train/, valid/, test/ with a data.yaml defining the single class: pothole.

## Model
YOLO11n — a single-stage, anchor-free CNN detector (backbone, neck, head). Started from the pretrained yolo11n.pt checkpoint and fine-tuned on the pothole dataset (transfer learning).

## Training
Trained for 100 epochs, imgsz=640, batch=16, on GPU (device=0). Each epoch: predict boxes, compare to labels, compute loss (box, class, boundary), backpropagate, update weights. Best-performing epoch saved as best.pt.

## Results (100 epochs)
- Precision: ~0.60
- Recall: ~0.48
- mAP50: ~0.51
- mAP50-95: ~0.23

## Inference
Video is decoded into frames (OpenCV), resized/padded to 640x640, passed through the model, filtered by confidence, deduplicated (NMS), boxes rescaled to original size, drawn on frames, and reassembled into an output video.

## Key Algorithms
- Gradient Descent + Backpropagation (training)
- Anchor-free detection
- CIoU / BCE / DFL losses
- Non-Max Suppression (NMS)

## Next Steps
- More/varied training data, more epochs, or a larger model (yolo11s/m)
- Web deployment (FastAPI) for remote video uploads - not yet built

## References
- Ultralytics Docs: https://docs.ultralytics.com
- Ultralytics GitHub: https://github.com/ultralytics/ultralytics
- PyTorch: https://github.com/pytorch/pytorch
- Roboflow SDK: https://github.com/roboflow/roboflow-python
- OpenCV: https://github.com/opencv/opencv-python
