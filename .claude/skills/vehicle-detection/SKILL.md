---
name: vehicle-detection
description: >
  Detect vehicles (car, bike, truck, bus) in video frames or images using YOLOv8 pretrained on the COCO dataset.
  Use this skill whenever the user wants to detect vehicles in traffic footage, CCTV streams, images, or video files.
  Triggers for: "detect vehicles", "find cars in video", "identify trucks/buses/bikes", "run YOLO on traffic footage",
  "vehicle detection model", "object detection for traffic", "COCO vehicle classes". Always use this skill before
  tracking or counting — detection is the first stage of the ANPR pipeline.
---

# Vehicle Detection Skill

Detect vehicles in real-time video streams or image frames using **YOLOv8** pretrained on the **COCO dataset**.

## Supported Vehicle Classes

| COCO Class | Label   |
| ---------- | ------- |
| car        | `car`   |
| motorcycle | `bike`  |
| truck      | `truck` |
| bus        | `bus`   |

---

## Dependencies

```bash
pip install ultralytics opencv-python
```

---

## Step-by-Step Implementation

### 1. Load YOLOv8 Model

```python
from ultralytics import YOLO
import cv2

# Load pretrained YOLOv8 (nano for speed, or yolov8m/l for accuracy)
model = YOLO("yolov8n.pt")  # Downloads automatically on first run

# COCO class IDs for vehicles
VEHICLE_CLASSES = {
    2: "car",
    3: "bike",       # motorcycle in COCO
    5: "bus",
    7: "truck"
}
```

### 2. Process Video Frames

```python
def detect_vehicles(frame, model, conf_threshold=0.4):
    """
    Run YOLOv8 detection on a single frame.
    Returns list of detections: [x1, y1, x2, y2, confidence, class_id, class_name]
    """
    results = model(frame, conf=conf_threshold, verbose=False)[0]
    detections = []

    for box in results.boxes:
        class_id = int(box.cls[0])
        if class_id in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_name = VEHICLE_CLASSES[class_id]
            detections.append([x1, y1, x2, y2, confidence, class_id, class_name])

    return detections
```

### 3. Draw Bounding Boxes

```python
COLORS = {"car": (0, 255, 0), "bike": (255, 165, 0), "truck": (0, 0, 255), "bus": (255, 0, 255)}

def draw_detections(frame, detections):
    for (x1, y1, x2, y2, conf, cls_id, cls_name) in detections:
        color = COLORS.get(cls_name, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame
```

### 4. Full Video Loop

```python
def run_vehicle_detection(video_path: str, output_path: str = None):
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_vehicles(frame, model)
        frame = draw_detections(frame, detections)

        if output_path:
            writer.write(frame)

    cap.release()
    if output_path:
        writer.release()
    print("Detection complete.")
```

---

## Model Selection Guide

| Model      | Speed   | Accuracy | Recommended For          |
| ---------- | ------- | -------- | ------------------------ |
| yolov8n.pt | Fastest | Lower    | Real-time / edge devices |
| yolov8s.pt | Fast    | Medium   | Balanced performance     |
| yolov8m.pt | Medium  | High     | Server-side processing   |
| yolov8l.pt | Slower  | Higher   | Best accuracy            |

---

## Output Format

Each detection returns:

```
[x1, y1, x2, y2, confidence, class_id, class_name]
```

Pass `detections` to the **vehicle-tracking** skill for ID assignment.

---

## Notes

- COCO class 3 is `motorcycle`, mapped to `bike` for this system.
- Set `conf_threshold=0.4` minimum; increase to 0.5–0.6 to reduce false positives.
- For license plate detection after vehicle detection, see the **license-plate-ocr** skill.
- For the full pipeline, see the **anpr-pipeline** skill.
