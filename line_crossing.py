import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

# Patch torch.load for PyTorch 2.6+ compatibility
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

from ultralytics import YOLO


# COCO vehicle classes
VEHICLE_CLASSES = {
    2: "car",
    3: "motorbike",
    5: "bus",
    7: "truck"
}

TRACK_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
]


def calculate_line_eq(p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[float, float, float]:
    """
    Calculate line equation Ax + By + C = 0 from two points.
    Returns (A, B, C) coefficients.
    """
    x1, y1 = p1
    x2, y2 = p2
    # Line: (y1-y2)*x + (x2-x1)*y + (x1*y2 - x2*y1) = 0
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    return A, B, C


def point_to_line_distance(A: float, B: float, C: float,
                          px: int, py: int) -> float:
    """Calculate perpendicular distance from point to line."""
    return abs(A * px + B * py + C) / np.sqrt(A * A + B * B)


def cross_product_2d(p1: Tuple[int, int], p2: Tuple[int, int],
                      p3: Tuple[int, int]) -> float:
    """Calculate 2D cross product (p1->p2) x (p1->p3). Positive = counter-clockwise."""
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])


class SimpleTracker:
    """Simple IoU-based tracker."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 1

    def box_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def update(self, detections: List[Dict], frame_idx: int = 0):
        matched_ids = set()
        tracked = []

        for det in detections:
            best_iou, best_id = 0, None
            for tid, tdata in self.tracks.items():
                if tid in matched_ids:
                    continue
                if tdata["class_name"] != det["class_name"]:
                    continue
                iou = self.box_iou(det["bbox"], tdata["bbox"])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_id = tid

            if best_id is not None:
                self.tracks[best_id] = {"bbox": det["bbox"], "class_name": det["class_name"], "age": 0}
                matched_ids.add(best_id)
                det["track_id"] = best_id
            else:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {"bbox": det["bbox"], "class_name": det["class_name"], "age": 0}
                det["track_id"] = new_id

            tracked.append(det)

        dead = [tid for tid, d in self.tracks.items() if d["age"] > self.max_age]
        for tid in dead:
            del self.tracks[tid]

        return tracked


class CrossingLine:
    """
    Virtual line for detecting vehicle crossings.
    """

    def __init__(self, p1: Tuple[int, int], p2: Tuple[int, int],
                 direction: str = "both"):
        """
        Initialize crossing line.

        Args:
            p1: Start point (x, y)
            p2: End point (x, y)
            direction: "up", "down", "both" - which direction counts
        """
        self.p1 = p1
        self.p2 = p2
        self.direction = direction
        self.A, self.B, self.C = calculate_line_eq(p1, p2)

    def check_crossing(self, prev_pos: Tuple[int, int],
                      curr_pos: Tuple[int, int]) -> Optional[str]:
        """
        Check if a point crossed the line from previous to current position.

        Returns:
            "up", "down", or None if no crossing
        """
        # Check if points are on different sides of the line
        prev_side = self.A * prev_pos[0] + self.B * prev_pos[1] + self.C
        curr_side = self.A * curr_pos[0] + self.B * curr_pos[1] + self.C

        # Same side = no crossing
        if prev_side * curr_side >= 0:
            return None

        # Determine direction based on which way we crossed
        # For a line from left to right: negative cross = up, positive = down
        cp = cross_product_2d(self.p1, self.p2, curr_pos)

        if cp > 0:
            direction = "down"
        else:
            direction = "up"

        if self.direction == "both" or self.direction == direction:
            return direction

        return None

    def draw(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 255),
             thickness: int = 3) -> np.ndarray:
        """Draw the crossing line on frame."""
        output = frame.copy()
        cv2.line(output, self.p1, self.p2, color, thickness)

        # Draw direction arrow
        mid_x = (self.p1[0] + self.p2[0]) // 2
        mid_y = (self.p1[1] + self.p2[1]) // 2

        # Perpendicular offset for arrow
        length = 20
        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        norm = np.sqrt(dx * dx + dy * dy)
        if norm > 0:
            nx, ny = -dy / norm, dx / norm
            # Arrow direction depending on crossing direction
            if self.direction in ("both", "down"):
                cv2.line(output, (mid_x, mid_y),
                        (int(mid_x - nx * length), int(mid_y - ny * length)),
                        color, 2)
            else:
                cv2.line(output, (mid_x, mid_y),
                        (int(mid_x + nx * length), int(mid_y + ny * length)),
                        color, 2)

        return output


class VehicleLineCounter:
    """
    Vehicle counter using line crossing detection.
    """

    def __init__(self,
                 line: CrossingLine,
                 model_path: str = None,
                 confidence: float = 0.3):
        self.line = line
        self.model = YOLO(model_path or "yolov8n.pt")
        self.confidence = confidence
        self.tracker = SimpleTracker()

        # Crossing history: track_id -> last position
        self.track_positions = {}

        # Counters
        self.counts = defaultdict(int)  # Total counts by direction and class
        self.crossings = []  # List of crossing events

    def reset(self):
        """Reset counter state."""
        self.tracker = SimpleTracker()
        self.track_positions.clear()
        self.counts.clear()
        self.crossings.clear()

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Detect vehicles in frame."""
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
        """Update with new frame and detect crossings."""
        detections = self.detect(frame)
        tracked = self.tracker.update(detections, frame_idx)

        # Convert to dict for position tracking
        current_positions = {t["track_id"]: t["center"] for t in tracked}

        # Check for crossings
        new_crossings = []

        for track in tracked:
            track_id = track["track_id"]
            class_name = track["class_name"]
            curr_pos = track["center"]

            if track_id in self.track_positions:
                prev_pos = self.track_positions[track_id]
                direction = self.line.check_crossing(prev_pos, curr_pos)

                if direction:
                    # Crossing detected!
                    event = {
                        "track_id": track_id,
                        "class_name": class_name,
                        "direction": direction,
                        "frame_idx": frame_idx,
                        "center": curr_pos
                    }
                    new_crossings.append(event)
                    self.crossings.append(event)
                    self.counts[f"{direction}_{class_name}"] += 1

            # Update position
            self.track_positions[track_id] = curr_pos

        return tracked, new_crossings

    def draw(self, frame: np.ndarray, tracks: list,
             new_crossings: list = None) -> np.ndarray:
        """Draw tracks and crossings on frame."""
        # Draw crossing line
        output = self.line.draw(frame)

        # Draw tracks
        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            track_id = track["track_id"]
            color = TRACK_COLORS[track_id % len(TRACK_COLORS)]

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label = f"ID {track_id}: {track['class_name']}"
            cv2.putText(output, label, (x1, y1 - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Highlight new crossings
        if new_crossings:
            for cross in new_crossings:
                cx, cy = cross["center"]
                color = (0, 255, 0) if cross["direction"] == "down" else (0, 0, 255)
                cv2.circle(output, (cx, cy), 10, color, -1)
                cv2.putText(output, f"{cross['class_name']}!",
                          (cx + 15, cy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                          0.6, color, 2)

        return output

    def get_counts(self, direction: str = None) -> Dict[str, int]:
        """Get crossing counts."""
        if direction:
            return {k: v for k, v in self.counts.items() if k.startswith(direction)}
        return dict(self.counts)


def process_video(
    video_path: str,
    line_p1: Tuple[int, int],
    line_p2: Tuple[int, int],
    output_path: str = None,
    frame_interval: int = 1,
    direction: str = "both",
    confidence: float = 0.3,
    show: bool = False
) -> Dict[str, Any]:
    """Process video with line crossing detection."""

    # Create crossing line
    line = CrossingLine(line_p1, line_p2, direction)
    counter = VehicleLineCounter(line, confidence=confidence)

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
            tracks, crossings = counter.update(frame, frame_idx)
            output_frame = counter.draw(frame, tracks, crossings)

            if writer:
                writer.write(output_frame)

            if show:
                cv2.imshow("Line Crossing", output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Print crossing events
            for cross in crossings:
                print(f"Frame {frame_idx}: {cross['class_name']} crossed {cross['direction']}")

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    return {
        "frames_processed": frame_idx,
        "total_crossings": len(counter.crossings),
        "counts": counter.get_counts()
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect vehicle line crossings")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--p1", nargs=2, type=int, default=[320, 240],
                        metavar=("X", "Y"), help="Line start point")
    parser.add_argument("--p2", nargs=2, type=int, default=[960, 240],
                        metavar=("X", "Y"), help="Line end point")
    parser.add_argument("--dir", default="both",
                        choices=["up", "down", "both"], help="Crossing direction")
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
        args.output, args.interval, args.dir,
        args.confidence, args.show
    )

    print(f"\nLine Crossing Results:")
    print(f"  Frames processed: {result['frames_processed']}")
    print(f"  Total crossings: {result['total_crossings']}")
    print(f"  Counts:")
    for key, count in result['counts'].items():
        direction, class_name = key.split("_", 1)
        print(f"    {class_name} ({direction}): {count}")