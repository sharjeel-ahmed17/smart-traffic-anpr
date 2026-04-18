"""
ANPR Pipeline: Full traffic monitoring with license plate detection on line crossing.
Integrates: Detection → Tracking → Line Crossing → Plate Detection → OCR
"""

import cv2
import numpy as np
import torch
import os
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from collections import defaultdict

# Patch torch.load for PyTorch 2.6+ compatibility
_original_torch_load = torch.load

def _patched_torch_load(*a, **kw):
    kw.setdefault('weights_only', False)
    return _original_torch_load(*a, **kw)

torch.load = _patched_torch_load

from ultralytics import YOLO


# COCO vehicle classes
VEHICLE_CLASSES = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}

TRACK_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
]


# ============ Simple Tracker (from line_crossing.py) ============

class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_age=30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 1

    def box_iou(self, b1, b2):
        x1_1, y1_1, x2_1, y2_1 = b1
        x1_2, y1_2, x2_2, y2_2 = b2
        xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
        xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        return inter / (area1 + area2 - inter) if area1 + area2 > inter else 0

    def update(self, detections, frame_idx=0):
        matched, tracked = set(), []
        for det in detections:
            best_iou, best_id = 0, None
            for tid, tdata in self.tracks.items():
                if tid not in matched and tdata["class_name"] == det["class_name"]:
                    iou = self.box_iou(det["bbox"], tdata["bbox"])
                    if iou > best_iou and iou >= self.iou_threshold:
                        best_iou, best_id = iou, tid

            if best_id:
                self.tracks[best_id] = {"bbox": det["bbox"], "class_name": det["class_name"], "age": 0}
                matched.add(best_id)
                det["track_id"] = best_id
            else:
                det["track_id"] = self.next_id
                self.tracks[self.next_id] = {"bbox": det["bbox"], "class_name": det["class_name"], "age": 0}
                self.next_id += 1
            tracked.append(det)

        for tid in list(self.tracks.keys()):
            self.tracks[tid]["age"] += 1
            if self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]
        return tracked


# ============ Crossing Line ============

def calc_line_eq(p1, p2):
    x1, y1, x2, y2 = *p1, *p2
    return (y1 - y2, x2 - x1, x1 * y2 - x2 * y1)

def cross_product(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])


class CrossingLine:
    def __init__(self, p1, p2, direction="both"):
        self.p1, self.p2, self.direction = p1, p2, direction
        self.A, self.B, self.C = calc_line_eq(p1, p2)

    def check_crossing(self, prev_pos, curr_pos):
        prev_side = self.A * prev_pos[0] + self.B * prev_pos[1] + self.C
        curr_side = self.A * curr_pos[0] + self.B * curr_pos[1] + self.C
        if prev_side * curr_side >= 0:
            return None
        direction = "down" if cross_product(self.p1, self.p2, curr_pos) > 0 else "up"
        if self.direction in ("both", direction):
            return direction
        return None

    def draw(self, frame, color=(0, 255, 255), thickness=3):
        output = frame.copy()
        cv2.line(output, self.p1, self.p2, color, thickness)
        return output


# ============ Plate Detector ============

class PlateDetector:
    def __init__(self, model_path=None, confidence=0.3):
        self.confidence = confidence
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        elif os.path.exists("models/plate_detector.pt"):
            self.model = YOLO("models/plate_detector.pt")
        else:
            print("Using general YOLOv8 for plate detection")
            self.model = YOLO("yolov8n.pt")

    def detect(self, frame, roi=None):
        if roi:
            x1, y1, x2, y2 = roi
            frame = frame[y1:y2, x1:x2]
            detections = self.model(frame, conf=self.confidence, verbose=False)
            results = []
            for r in detections:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    bx1, by1, bx2, by2 = map(int, xyxy)
                    results.append({
                        "confidence": float(box.conf[0]),
                        "bbox": [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1],
                        "center": (int((bx1 + bx2) / 2) + x1, int((by1 + by2) / 2) + y1),
                        "class_name": "plate"
                    })
            return results
        else:
            results = []
            detections = self.model(frame, conf=self.confidence, verbose=False)
            for r in detections:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    bx1, by1, bx2, by2 = map(int, xyxy)
                    results.append({
                        "confidence": float(box.conf[0]),
                        "bbox": [bx1, by1, bx2, by2],
                        "center": (int((bx1 + bx2) / 2), int((by1 + by2) / 2)),
                        "class_name": "plate"
                    })
            return results


# ============ ANPR Pipeline ============

class ANPRPipeline:
    """
    Complete ANPR pipeline that:
    1. Detects vehicles in video frames
    2. Tracks vehicles across frames
    3. Detects when vehicles cross a virtual line
    4. Captures license plates for crossing vehicles
    """

    def __init__(self,
                 line_p1: Tuple[int, int],
                 line_p2: Tuple[int, int],
                 vehicle_model_path: str = None,
                 plate_model_path: str = None,
                 vehicle_confidence: float = 0.3,
                 plate_confidence: float = 0.3,
                 output_dir: str = "output/captured"):
        """
        Initialize ANPR pipeline.

        Args:
            line_p1: Start point of crossing line
            line_p2: End point of crossing line
            vehicle_model_path: Path to vehicle YOLO model
            plate_model_path: Path to plate detection model
            vehicle_confidence: Vehicle detection confidence
            plate_confidence: Plate detection confidence
            output_dir: Directory to save captured plates
        """
        # Vehicle detection model
        self.vehicle_model = YOLO(vehicle_model_path or "yolov8n.pt")
        self.vehicle_confidence = vehicle_confidence

        # Plate detection model
        self.plate_detector = PlateDetector(plate_model_path, plate_confidence)

        # Tracking
        self.tracker = SimpleTracker()

        # Crossing line
        self.line = CrossingLine(line_p1, line_p2, "both")

        # State
        self.track_positions = {}
        self.counts = defaultdict(int)
        self.crossings = []
        self.captured_plates = []

        # Output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def reset(self):
        """Reset pipeline state."""
        self.tracker = SimpleTracker()
        self.track_positions.clear()
        self.counts.clear()
        self.crossings.clear()
        self.captured_plates.clear()

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0,
                    capture_on_crossing: bool = True) -> Tuple[list, list, list]:
        """
        Process a single frame.

        Args:
            frame: Video frame (BGR)
            frame_idx: Current frame index
            capture_on_crossing: Whether to capture plates on crossing

        Returns:
            Tuple of (tracked_vehicles, crossings, plate_detections)
        """
        # Detect vehicles
        results = self.vehicle_model(frame, conf=self.vehicle_confidence, verbose=False)
        detections = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue

                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)

                detections.append({
                    "class_id": cls_id,
                    "class_name": VEHICLE_CLASSES[cls_id],
                    "confidence": float(box.conf[0]),
                    "bbox": [x1, y1, x2, y2],
                    "center": (int((x1 + x2) / 2), int((y1 + y2) / 2))
                })

        # Track vehicles
        tracked = self.tracker.update(detections, frame_idx)

        # Add colors
        for t in tracked:
            t["color"] = TRACK_COLORS[t["track_id"] % len(TRACK_COLORS)]

        # Check crossings
        new_crossings = []
        current_positions = {}

        for track in tracked:
            track_id = track["track_id"]
            class_name = track["class_name"]
            curr_pos = track["center"]
            current_positions[track_id] = curr_pos

            if track_id in self.track_positions:
                prev_pos = self.track_positions[track_id]
                direction = self.line.check_crossing(prev_pos, curr_pos)

                if direction:
                    event = {
                        "track_id": track_id,
                        "class_name": class_name,
                        "direction": direction,
                        "frame_idx": frame_idx,
                        "center": curr_pos,
                        "bbox": track["bbox"]
                    }
                    new_crossings.append(event)
                    self.crossings.append(event)
                    self.counts[f"{direction}_{class_name}"] += 1

                    # Capture license plate on crossing
                    if capture_on_crossing:
                        plate = self._capture_plate(frame, track["bbox"], track_id, frame_idx)
                        if plate:
                            self.captured_plates.append(plate)

            self.track_positions[track_id] = curr_pos

        return tracked, new_crossings, []

    def _capture_plate(self, frame, vehicle_bbox, track_id, frame_idx):
        """Capture license plate from vehicle region."""
        x1, y1, x2, y2 = vehicle_bbox

        # Add padding around vehicle
        h, w = frame.shape[:2]
        x1_p = max(0, x1 - 20)
        y1_p = max(0, y1 - 20)
        x2_p = min(w, x2 + 20)
        y2_p = min(h, y2 + 20)

        roi = (x1_p, y1_p, x2_p, y2_p)
        plates = self.plate_detector.detect(frame, roi)

        if plates:
            plate = plates[0]
            crop = frame[y1_p:y2_p, x1_p:x2_p]

            # Save captured plate
            filename = f"plate_{track_id}_{frame_idx:05d}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, crop)

            return {
                "track_id": track_id,
                "frame_idx": frame_idx,
                "plate_bbox": plate["bbox"],
                "filepath": filepath,
                "timestamp": datetime.now().isoformat()
            }
        return None

    def draw(self, frame, tracked, new_crossings):
        """Draw tracking visualization."""
        # Draw line
        output = self.line.draw(frame)

        # Draw vehicles
        for track in tracked:
            x1, y1, x2, y2 = track["bbox"]
            color = track["color"]
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label = f"ID {track['track_id']}: {track['class_name']}"
            cv2.putText(output, label, (x1, y1 - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Highlight crossings
        for cross in new_crossings:
            cx, cy = cross["center"]
            col = (0, 255, 0) if cross["direction"] == "down" else (0, 0, 255)
            cv2.circle(output, (cx, cy), 10, col, -1)

        return output

    def get_counts(self):
        """Get crossing counts."""
        return dict(self.counts)


def process_video(
    video_path: str,
    line_p1: Tuple[int, int],
    line_p2: Tuple[int, int],
    output_path: str = None,
    frame_interval: int = 1,
    confidence: float = 0.3,
    show: bool = False
) -> Dict[str, Any]:
    """Process video with full ANPR pipeline."""

    pipeline = ANPRPipeline(line_p1, line_p2, vehicle_confidence=confidence, plate_confidence=confidence)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps // frame_interval, (width, height))

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            tracked, crossings, _ = pipeline.process_frame(frame, frame_idx)

            output_frame = pipeline.draw(frame, tracked, crossings)

            if writer:
                writer.write(output_frame)

            if show:
                cv2.imshow("ANPR Pipeline", output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            for cross in crossings:
                print(f"Frame {frame_idx}: {cross['class_name']} (ID {cross['track_id']}) crossed {cross['direction']}")

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    return {
        "frames_processed": frame_idx,
        "total_crossings": len(pipeline.crossings),
        "plates_captured": len(pipeline.captured_plates),
        "counts": pipeline.get_counts()
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ANPR Pipeline - Full traffic monitoring")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--p1", nargs=2, type=int, default=[384, 0],
                      metavar=("X", "Y"), help="Line start point")
    parser.add_argument("--p2", nargs=2, type=int, default=[384, 432],
                      metavar=("X", "Y"), help="Line end point")
    parser.add_argument("--output", "-o", help="Output video path")
    parser.add_argument("--interval", "-i", type=int, default=1,
                      help="Process every N frames")
    parser.add_argument("--confidence", "-c", type=float, default=0.3,
                      help="Confidence threshold")
    parser.add_argument("--show", action="store_true", help="Display output")
    args = parser.parse_args()

    result = process_video(
        args.video,
        tuple(args.p1), tuple(args.p2),
        args.output, args.interval,
        args.confidence, args.show
    )

    print(f"\nANPR Pipeline Results:")
    print(f"  Frames processed: {result['frames_processed']}")
    print(f"  Total crossings: {result['total_crossings']}")
    print(f"  Plates captured: {result['plates_captured']}")
    print(f"  Counts:")
    for key, count in result['counts'].items():
        direction, class_name = key.split("_", 1)
        print(f"    {class_name} ({direction}): {count}")