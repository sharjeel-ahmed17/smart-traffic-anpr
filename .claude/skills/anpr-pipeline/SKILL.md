---
name: anpr-pipeline
description: >
  End-to-end Smart City AI Traffic Monitoring pipeline: video input → YOLOv8 vehicle detection →
  DeepSORT/ByteTrack tracking → virtual line crossing → license plate OCR → SQLite database → 
  Streamlit dashboard. Use this skill as the master orchestrator whenever the user wants to:
  run the full ANPR system, build the complete traffic monitoring pipeline, set up the smart traffic
  project from scratch, integrate all components (detection + tracking + OCR + DB + dashboard),
  create requirements.txt or README for the project, or understand how all parts connect.
  Also triggers for: "smart city traffic project", "vehicle counting ANPR", "full pipeline setup".
---

# Smart City ANPR Pipeline — Master Skill

End-to-end orchestration of the complete AI-powered traffic monitoring system.

## Architecture Overview

```
Video Input (CCTV / .mp4)
        ↓
  Frame Extraction (OpenCV)
        ↓
  YOLOv8 Vehicle Detection    ← vehicle-detection skill
        ↓
  DeepSORT / ByteTrack        ← vehicle-tracking skill
        ↓
  Line Crossing Detection     ← line-crossing-counter skill
        ↓
  YOLO License Plate + OCR    ← license-plate-ocr skill
        ↓
  SQLite / PostgreSQL         ← traffic-database skill
        ↓
  Streamlit Dashboard         ← traffic-dashboard skill
```

---

## Project Structure

```
smart-traffic-anpr/
├── data/
│   ├── videos/
│   │   └── traffic.mp4
│   └── images/
├── notebooks/
│   ├── vehicle_detection.ipynb
│   ├── tracking_and_counting.ipynb
│   └── license_plate_ocr.ipynb
├── models/
│   ├── vehicle_model.pt       ← YOLOv8 (COCO pretrained)
│   └── plate_model.pt         ← YOLOv8 (plate-trained)
├── database/
│   ├── traffic.db
│   └── schema.sql
├── app/
│   └── streamlit_app.py
├── output/
│   ├── results_video.mp4
│   └── logs.csv
├── requirements.txt
└── README.md
```

---

## requirements.txt

```
ultralytics>=8.0.0
opencv-python>=4.8.0
easyocr>=1.7.0
deep-sort-realtime>=1.3.2
supervision>=0.18.0
streamlit>=1.30.0
fastapi>=0.110.0
uvicorn>=0.27.0
pandas>=2.0.0
plotly>=5.18.0
Pillow>=10.0.0
pytesseract>=0.3.10
numpy>=1.24.0
```

---

## Main Pipeline Script (`main.py`)

```python
import cv2
import time
from datetime import datetime
from ultralytics import YOLO

# Import per-skill modules
from vehicle_detection  import detect_vehicles, draw_detections, VEHICLE_CLASSES
from vehicle_tracking   import init_bytetrack, track_vehicles_bytetrack, draw_tracks
from line_crossing      import check_line_crossing, draw_line_and_counts
from license_plate_ocr  import detect_and_read_plate
from traffic_database   import init_db, store_crossing

# ─── Config ────────────────────────────────────────────────
VIDEO_PATH   = "data/videos/traffic.mp4"
OUTPUT_PATH  = "output/results_video.mp4"
LINE_Y       = 400          # Adjust to your video
CONF_THRESH  = 0.4
OCR_ENGINE   = "easyocr"    # or "tesseract"

# ─── Init ──────────────────────────────────────────────────
vehicle_model = YOLO("models/vehicle_model.pt")
plate_model   = YOLO("models/plate_model.pt")
tracker       = init_bytetrack()
init_db()

prev_centroids = {}
crossed_ids    = set()
vehicle_counts = {"car": 0, "bike": 0, "truck": 0, "bus": 0}

# ─── Video I/O ─────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

print("▶ Processing video...")
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # ── 1. Detect vehicles ──────────────────────────────────
    detections = detect_vehicles(frame, vehicle_model, CONF_THRESH)

    # ── 2. Track vehicles ───────────────────────────────────
    tracked = track_vehicles_bytetrack(tracker, detections, frame.shape)

    # ── 3. Line crossing + ANPR trigger ─────────────────────
    newly_crossed = []
    for (x1, y1, x2, y2, track_id, cls_name) in tracked:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if check_line_crossing(track_id, cx, cy, LINE_Y, prev_centroids, crossed_ids):
            vehicle_counts[cls_name] = vehicle_counts.get(cls_name, 0) + 1
            crossed_ids.add(track_id)
            newly_crossed.append({
                "track_id":   track_id,
                "class_name": cls_name,
                "bbox":       (x1, y1, x2, y2),
            })
        prev_centroids[track_id] = (cx, cy)

    # ── 4. OCR for crossing vehicles ────────────────────────
    for vehicle in newly_crossed:
        plate_text = detect_and_read_plate(
            frame, vehicle["bbox"], plate_model, OCR_ENGINE
        )
        store_crossing(
            vehicle_type  = vehicle["class_name"],
            plate_number  = plate_text,
            direction     = "down",
            track_id      = vehicle["track_id"]
        )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"{vehicle['class_name'].upper()} | Plate: {plate_text}")

    # ── 5. Annotate frame ───────────────────────────────────
    frame = draw_tracks(frame, tracked)
    frame = draw_line_and_counts(frame, LINE_Y, vehicle_counts)

    writer.write(frame)

cap.release()
writer.release()

print(f"\n✅ Done! Processed {frame_count} frames.")
print(f"📊 Final Counts: {vehicle_counts}")
print(f"📹 Output saved: {OUTPUT_PATH}")
print(f"🗃️  Logs in:     database/traffic.db")
```

---

## Run Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the pipeline
python main.py

# 3. Launch dashboard
streamlit run app/streamlit_app.py

# 4. (Optional) Launch API
uvicorn app.api:app --reload --port 8000
```

---

## Skill Reference Map

| Task                           | Skill to Read             |
| ------------------------------ | ------------------------- |
| Detect cars/trucks/buses/bikes | **vehicle-detection**     |
| Assign unique IDs per vehicle  | **vehicle-tracking**      |
| Count vehicles at a line       | **line-crossing-counter** |
| Read license plate text        | **license-plate-ocr**     |
| Store and query event logs     | **traffic-database**      |
| Build analytics dashboard      | **traffic-dashboard**     |

---

## Dataset Links

- **License Plate Dataset (Roboflow):**  
  `https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e`

- **Sample Test Videos (Google Drive):**  
  `https://drive.google.com/drive/folders/11n5u1B3BAppISJl7OgLNII3shY9FnLsV?usp=sharing`

---

## Expected Output

```
Vehicle Type | Plate Number | Timestamp
-------------|--------------|--------------------
Truck        | ABC-123      | 10:01 AM
Truck        | XYZ-456      | 10:02 AM
Truck        | DEF-789      | 10:05 AM
```

Dashboard shows: live class-wise counts, plate log table, hourly charts, CSV export.

---

## Impact

This system helps smart cities automate traffic monitoring, reduce manual surveillance costs,
and improve law enforcement efficiency through real-time ANPR and analytics.
