import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from collections import defaultdict

# Patch torch.load for PyTorch 2.6+ compatibility
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

from ultralytics import YOLO


# COCO class IDs for vehicle types
VEHICLE_CLASSES = {
    2: "car",
    3: "motorbike",
    5: "bus",
    7: "truck"
}

# Unique colors for different track IDs
TRACK_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (128, 255, 0), (255, 0, 128), (128, 0, 255)
]


def box_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Intersection area
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


class SimpleTracker:
    """Simple IoU-based tracker for vehicle tracking."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = {}  # track_id -> {"bbox": [...], "class_name": str, "age": int}
        self.next_id = 1
        self.active_ids = set()

    def update(self, detections: List[Dict], frame_idx: int = 0):
        """
        Update tracker with new detections.

        Args:
            detections: List of detection dicts with "bbox" and "class_name"
            frame_idx: Current frame index

        Returns:
            List of tracked detections with assigned track_ids
        """
        # Age existing tracks
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]["age"] += 1

        # Match detections to tracks
        matched_ids = set()
        tracked = []

        for det in detections:
            best_iou = 0
            best_track_id = None

            for track_id, track_data in self.tracks.items():
                if track_id in matched_ids:
                    continue
                if track_data["class_name"] != det["class_name"]:
                    continue

                iou = box_iou(det["bbox"], track_data["bbox"])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                # Update existing track
                self.tracks[best_track_id] = {
                    "bbox": det["bbox"],
                    "class_name": det["class_name"],
                    "age": 0
                }
                matched_ids.add(best_track_id)
                det["track_id"] = best_track_id
            else:
                # Create new track
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {
                    "bbox": det["bbox"],
                    "class_name": det["class_name"],
                    "age": 0
                }
                det["track_id"] = new_id

            tracked.append(det)

        # Remove old tracks
        dead_ids = [tid for tid, data in self.tracks.items() if data["age"] > self.max_age]
        for tid in dead_ids:
            del self.tracks[tid]

        self.active_ids = set(t["track_id"] for t in tracked)
        return tracked


class VehicleTracker:
    """
    YOLOv8-based vehicle tracker.
    Uses simple IoU matching for tracking across frames.
    """

    def __init__(self, model_path: str = None, confidence: float = 0.3,
                 iou_threshold: float = 0.3, max_age: int = 30):
        self.model = YOLO(model_path or "yolov8n.pt")
        self.confidence = confidence
        self.tracker = SimpleTracker(iou_threshold, max_age)
        self.seen_ids = set()
        self.frame_count = 0

    def reset(self):
        """Reset tracker state."""
        self.tracker = SimpleTracker()
        self.seen_ids = set()
        self.frame_count = 0

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Detect vehicles without tracking."""
        results = self.model(frame, conf=self.confidence, verbose=False)
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

        return detections

    def update(self, frame: np.ndarray, frame_idx: int = 0):
        """Detect and track vehicles."""
        detections = self.detect(frame)
        tracked = self.tracker.update(detections, frame_idx)

        # Assign colors and track IDs
        for t in tracked:
            self.seen_ids.add(t["track_id"])
            t["color"] = TRACK_COLORS[t["track_id"] % len(TRACK_COLORS)]

        self.frame_count += 1
        return tracked

    def draw_tracks(self, frame: np.ndarray, tracks: list) -> np.ndarray:
        """Draw tracking boxes and IDs."""
        output = frame.copy()

        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            color = track["color"]
            track_id = track["track_id"]
            class_name = track["class_name"]

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label = f"ID {track_id}: {class_name}"
            cv2.putText(output, label, (x1, y1 - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cx, cy = track["center"]
            cv2.circle(output, (cx, cy), 4, color, -1)

        return output

    def get_counts(self) -> Dict[str, int]:
        """Get unique vehicle counts by class."""
        class_counts = defaultdict(int)
        for track_id, track_data in self.tracker.tracks.items():
            class_counts[track_data["class_name"]] += 1
        return dict(class_counts)


def process_video(
    video_path: str,
    output_path: str = None,
    frame_interval: int = 1,
    confidence: float = 0.3,
    show: bool = False
) -> Dict[str, Any]:
    """Process video with tracking."""
    tracker = VehicleTracker(confidence=confidence)
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
    total_detections = 0
    class_counts = defaultdict(int)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            tracks = tracker.update(frame, frame_idx)
            total_detections += len(tracks)

            for track in tracks:
                class_counts[track["class_name"]] += 1

            output_frame = tracker.draw_tracks(frame, tracks)

            if writer:
                writer.write(output_frame)

            if show:
                cv2.imshow("Vehicle Tracking", output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    return {
        "frames_processed": frame_idx,
        "total_detections": total_detections,
        "unique_vehicles": len(tracker.seen_ids),
        "class_counts": dict(class_counts)
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Track vehicles in video")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--output", "-o", help="Output video path")
    parser.add_argument("--interval", "-i", type=int, default=1,
                        help="Process every N frames")
    parser.add_argument("--confidence", "-c", type=float, default=0.3,
                        help="Confidence threshold")
    parser.add_argument("--show", action="store_true",
                        help="Display output")
    args = parser.parse_args()

    result = process_video(
        args.video,
        args.output,
        args.interval,
        args.confidence,
        args.show
    )

    print(f"\nTracking Results:")
    print(f"  Frames processed: {result['frames_processed']}")
    print(f"  Total detections: {result['total_detections']}")
    print(f"  Unique vehicles: {result['unique_vehicles']}")
    print(f"  Class counts:")
    for cls, count in result['class_counts'].items():
        print(f"    {cls}: {count}")