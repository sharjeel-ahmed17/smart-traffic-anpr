---
name: vehicle-tracking
description: >
  Track detected vehicles across video frames using DeepSORT or ByteTrack, assigning unique IDs
  to each vehicle so they are not double-counted. Use this skill whenever the user needs to:
  track vehicles across frames, assign persistent IDs, prevent duplicate counting, implement
  multi-object tracking (MOT), integrate DeepSORT or ByteTrack into a traffic pipeline.
  This skill takes YOLOv8 detections as input and must be used AFTER vehicle-detection and
  BEFORE line-crossing-counter in the ANPR pipeline.
---

# Vehicle Tracking Skill

Assign unique persistent IDs to detected vehicles across frames using **DeepSORT** or **ByteTrack**,
preventing the same vehicle from being counted multiple times.

---

## Dependencies

```bash
# Option A: ByteTrack (lightweight, recommended)
pip install supervision

# Option B: DeepSORT
pip install deep-sort-realtime
```

---

## Option A: ByteTrack via Supervision (Recommended)

ByteTrack is faster and simpler to set up. Use this unless the user specifically asks for DeepSORT.

```python
import supervision as sv
import numpy as np

def init_bytetrack():
    """Initialize ByteTrack tracker."""
    return sv.ByteTracker(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30
    )

def track_vehicles_bytetrack(tracker, detections_raw, frame_shape):
    """
    Convert raw YOLO detections to supervision Detections and update tracker.

    detections_raw: list of [x1, y1, x2, y2, confidence, class_id, class_name]
    Returns: list of [x1, y1, x2, y2, track_id, class_name]
    """
    if not detections_raw:
        return []

    boxes = np.array([[d[0], d[1], d[2], d[3]] for d in detections_raw])
    confidences = np.array([d[4] for d in detections_raw])
    class_ids = np.array([d[5] for d in detections_raw])
    class_names = [d[6] for d in detections_raw]

    sv_detections = sv.Detections(
        xyxy=boxes,
        confidence=confidences,
        class_id=class_ids
    )

    tracked = tracker.update_with_detections(sv_detections)

    results = []
    for i, box in enumerate(tracked.xyxy):
        x1, y1, x2, y2 = map(int, box)
        track_id = int(tracked.tracker_id[i])
        cls_id = int(tracked.class_id[i])
        # Map back to class name
        cls_name = class_names[cls_id] if cls_id < len(class_names) else "vehicle"
        results.append([x1, y1, x2, y2, track_id, cls_name])

    return results
```

---

## Option B: DeepSORT

Use when appearance-based re-identification is needed (e.g., vehicles reappear after occlusion).

```python
from deep_sort_realtime.deepsort_tracker import DeepSort

def init_deepsort():
    return DeepSort(
        max_age=30,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=0.3,
        nn_budget=None,
        embedder="mobilenet",
        half=True,
        bgr=True,
    )

def track_vehicles_deepsort(tracker, detections_raw, frame):
    """
    detections_raw: list of [x1, y1, x2, y2, confidence, class_id, class_name]
    Returns: list of [x1, y1, x2, y2, track_id, class_name]
    """
    if not detections_raw:
        return []

    # DeepSORT expects: [[left, top, width, height], confidence, class]
    ds_input = []
    for d in detections_raw:
        x1, y1, x2, y2, conf, cls_id, cls_name = d
        w, h = x2 - x1, y2 - y1
        ds_input.append(([x1, y1, w, h], conf, cls_name))

    tracks = tracker.update_tracks(ds_input, frame=frame)

    results = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        track_id = track.track_id
        cls_name = track.get_det_class() or "vehicle"
        results.append([x1, y1, x2, y2, track_id, cls_name])

    return results
```

---

## Draw Tracked Vehicles

```python
import cv2

def draw_tracks(frame, tracked_vehicles):
    """Draw bounding boxes with persistent track IDs."""
    COLORS = {
        "car": (0, 255, 0),
        "bike": (255, 165, 0),
        "truck": (0, 0, 255),
        "bus": (255, 0, 255)
    }
    for (x1, y1, x2, y2, track_id, cls_name) in tracked_vehicles:
        color = COLORS.get(cls_name, (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{track_id} {cls_name}"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame
```

---

## Compute Centroid

The centroid is used by the **line-crossing-counter** skill to detect when a vehicle crosses the virtual line.

```python
def get_centroid(x1, y1, x2, y2):
    """Return (cx, cy) center point of bounding box."""
    return ((x1 + x2) // 2, (y1 + y2) // 2)
```

---

## Tracker Comparison

| Feature               | ByteTrack     | DeepSORT             |
| --------------------- | ------------- | -------------------- |
| Speed                 | ⚡ Very fast  | Moderate             |
| Re-ID after occlusion | Basic         | Strong (appearance)  |
| Setup complexity      | Simple        | Needs embedder model |
| Best for              | Dense traffic | Sparse / re-entry    |

---

## Output Format

Each tracked vehicle:

```
[x1, y1, x2, y2, track_id, class_name]
```

Pass to **line-crossing-counter** skill to detect line crossings.

---

## Notes

- `track_id` is unique per vehicle session; it does NOT persist across video restarts.
- Increase `max_age` if vehicles temporarily disappear behind obstructions.
- For the full pipeline, see the **anpr-pipeline** skill.
