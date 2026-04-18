---
name: line-crossing-counter
description: >
  Define a virtual line on a road video and count vehicles that cross it, class by class (car, bike, truck, bus).
  Triggers license plate detection when a vehicle crosses. Use this skill whenever the user needs:
  virtual line detection, vehicle counting, crossing events, count cars/trucks/buses crossing a line,
  traffic flow counting, line trip-wire for ANPR, centroid-based line detection.
  Must be used AFTER vehicle-tracking. When a crossing is detected, trigger the license-plate-ocr skill.
---

# Line Crossing Counter Skill

Define a virtual line across the road and count vehicles per class when their centroid crosses it.
Crossing events also trigger license plate extraction.

---

## Concept

```
Frame:
  ________________________________________
 |                                        |
 |    [car ID:1]  →  →  →                |
 |                                        |
 |════════════════════════════════════════|  ← Virtual Line (y = LINE_Y)
 |                                        |
 |_______________________________________|
```

A vehicle **crosses** when its centroid moves from **above → below** (or below → above) the line.

---

## Dependencies

```bash
pip install opencv-python
```

---

## Step 1: Define the Virtual Line

```python
import cv2

# Define line as y-coordinate (horizontal line) or two points (angled line)
# For a simple horizontal line:
LINE_Y = 400   # pixel row — set based on your video resolution

# For a diagonal line use two points:
LINE_START = (0, 400)     # left edge
LINE_END   = (1280, 400)  # right edge (full width)
```

**How to choose LINE_Y:**

- Set it to roughly the middle-lower third of the frame.
- Avoid placing it too close to the edge (vehicles may be partially out of frame).
- Adjust after first run by visually inspecting the output video.

---

## Step 2: Track Previous Positions

```python
# Store last known centroid Y for each tracked vehicle ID
prev_centroids = {}   # {track_id: (cx, cy)}

# Vehicle counters per class
vehicle_counts = {
    "car": 0,
    "bike": 0,
    "truck": 0,
    "bus": 0
}

# Set of IDs that already crossed (prevent double counting)
crossed_ids = set()
```

---

## Step 3: Detect Crossing

```python
def check_line_crossing(track_id, cx, cy, line_y, prev_centroids, crossed_ids):
    """
    Returns True if this vehicle just crossed the line this frame.
    Crossing = centroid moved from one side to the other.
    """
    if track_id in crossed_ids:
        return False  # Already counted

    if track_id in prev_centroids:
        prev_cy = prev_centroids[track_id][1]
        # Crossed downward (top→bottom)
        if prev_cy < line_y <= cy:
            return True
        # Crossed upward (bottom→top) — optional, for bidirectional counting
        if prev_cy > line_y >= cy:
            return True

    return False
```

---

## Step 4: Full Frame Processing

```python
def process_frame_for_crossing(frame, tracked_vehicles, line_y,
                                prev_centroids, crossed_ids, vehicle_counts):
    """
    tracked_vehicles: list of [x1, y1, x2, y2, track_id, class_name]
    Returns: updated counts, list of newly crossed vehicles (for ANPR trigger)
    """
    newly_crossed = []   # [{track_id, class_name, bbox, frame}] → pass to OCR

    for (x1, y1, x2, y2, track_id, cls_name) in tracked_vehicles:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if check_line_crossing(track_id, cx, cy, line_y, prev_centroids, crossed_ids):
            # Count this vehicle
            if cls_name in vehicle_counts:
                vehicle_counts[cls_name] += 1
            crossed_ids.add(track_id)

            # Trigger ANPR for this vehicle
            newly_crossed.append({
                "track_id": track_id,
                "class_name": cls_name,
                "bbox": (x1, y1, x2, y2),
                "frame": frame.copy()
            })

        # Update last known centroid
        prev_centroids[track_id] = (cx, cy)

    return vehicle_counts, newly_crossed
```

---

## Step 5: Draw Line and Counts on Frame

```python
def draw_line_and_counts(frame, line_y, vehicle_counts):
    h, w = frame.shape[:2]

    # Draw virtual line
    cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
    cv2.putText(frame, "COUNT LINE", (10, line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Draw vehicle counts (top-left corner)
    y_offset = 30
    for cls_name, count in vehicle_counts.items():
        label = f"{cls_name.capitalize()}: {count}"
        cv2.putText(frame, label, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30

    return frame
```

---

## Step 6: ANPR Trigger

When `newly_crossed` is non-empty, pass each entry to the **license-plate-ocr** skill:

```python
from license_plate_ocr import detect_and_read_plate

for vehicle in newly_crossed:
    plate_text = detect_and_read_plate(vehicle["frame"], vehicle["bbox"])
    # Then store in DB via traffic-database skill
    store_event(vehicle["class_name"], plate_text, timestamp=now())
```

---

## Bidirectional Counting (Optional)

To count vehicles going in both directions:

```python
up_counts   = {"car": 0, "bike": 0, "truck": 0, "bus": 0}
down_counts = {"car": 0, "bike": 0, "truck": 0, "bus": 0}

def check_direction(prev_cy, cy, line_y):
    if prev_cy < line_y <= cy:
        return "down"
    if prev_cy > line_y >= cy:
        return "up"
    return None
```

---

## Notes

- `crossed_ids` ensures each vehicle is only counted once, even across multiple frames near the line.
- For angled roads, use a line defined by two points and compute the crossing with vector math.
- For the full pipeline, see the **anpr-pipeline** skill.
