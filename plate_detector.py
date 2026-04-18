import cv2
import numpy as np
import torch
import os
from typing import List, Dict, Any, Optional, Tuple

# Patch torch.load for PyTorch 2.6+ compatibility
_original_torch_load = torch.load

def _patched_torch_load(*a, **kw):
    kw.setdefault('weights_only', False)
    return _original_torch_load(*a, **kw)

torch.load = _patched_torch_load

from ultralytics import YOLO


# Default plate model URL (Ultralytics HUB)
DEFAULT_PLATE_MODEL = "yolov8n.pt"

# Try to load plate detection model, fall back to general detection
def download_plate_model(model_path: str = "models/plate_detector.pt") -> str:
    """
    Download or prepare license plate detection model.
    Uses a simple approach if no custom model available.
    """
    if os.path.exists(model_path):
        return model_path

    # Create models directory
    os.makedirs("models", exist_ok=True)

    # Download a pretrained plate detection model from Ultralytics HUB
    # Using a public model - license plate detection from Roboflow
    model_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"

    print("Note: No custom plate model found. Using YOLOv8n for general detection.")
    print("For better plate detection, place a trained plate model at: models/plate_detector.pt")
    print("You can train one at: https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e")

    return "yolov8n.pt"


class PlateDetector:
    """
    YOLO-based license plate detector.
    Can use a custom plate detection model or fallback to general detection.
    """

    def __init__(self, model_path: Optional[str] = None, confidence: float = 0.3):
        """
        Initialize license plate detector.

        Args:
            model_path: Path to plate detection model. If None, tries to use default.
            confidence: Minimum confidence threshold
        """
        self.confidence = confidence

        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # Try default model path
            default_path = "models/plate_detector.pt"
            if os.path.exists(default_path):
                self.model = YOLO(default_path)
            else:
                # Fallback to general YOLOv8 - will detect general objects
                print("Warning: Using general YOLOv8 model for plate detection.")
                print("Place a trained plate model at models/plate_detector.pt for better results.")
                self.model = YOLO("yolov8n.pt")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect license plates in a frame.

        Args:
            frame: BGR image frame from OpenCV

        Returns:
            List of detection dictionaries
        """
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)

                detections.append({
                    "confidence": float(box.conf[0]),
                    "bbox": [x1, y1, x2, y2],
                    "center": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                    "class_name": "plate"
                })

        return detections

    def crop_plate(self, frame: np.ndarray,
                   detection: Dict[str, Any]) -> np.ndarray:
        """
        Crop the license plate region from the frame.

        Args:
            frame: Original BGR frame
            detection: Detection dictionary with bbox

        Returns:
            Cropped plate image
        """
        x1, y1, x2, y2 = detection["bbox"]
        # Add small padding
        pad = 5
        h, w = frame.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        return frame[y1:y2, x1:x2]

    def detect_in_roi(self, frame: np.ndarray,
                     roi: Tuple[int, int, int, int]) -> List[Dict[str, Any]]:
        """
        Detect plates within a region of interest.

        Args:
            frame: Full frame
            roi: (x1, y1, x2, y2) region of interest

        Returns:
            List of detections within ROI
        """
        x1, y1, x2, y2 = roi
        roi_frame = frame[y1:y2, x1:x2]

        detections = self.detect(roi_frame)

        # Adjust bbox coordinates to full frame
        for det in detections:
            det["bbox"] = [
                det["bbox"][0] + x1,
                det["bbox"][1] + y1,
                det["bbox"][2] + x1,
                det["bbox"][3] + y1
            ]
            det["center"] = (
                det["center"][0] + x1,
                det["center"][1] + y1
            )

        return detections

    def draw_detections(self, frame: np.ndarray,
                       detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw plate detection bounding boxes.

        Args:
            frame: BGR frame
            detections: List of detection dicts

        Returns:
            Frame with drawn boxes
        """
        output = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = (0, 255, 255)  # Yellow for plates

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            label = f"Plate {det['confidence']:.2f}"
            cv2.putText(output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return output


# Simple IoU tracker for plates
class PlateTracker:
    """Simple tracker for license plates across frames."""

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
        matched = set()
        tracked = []

        for det in detections:
            best_iou, best_id = 0, None

            for tid, tdata in self.tracks.items():
                if tid in matched:
                    continue
                iou = self.box_iou(det["bbox"], tdata["bbox"])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_id = tid

            if best_id is not None:
                self.tracks[best_id] = {"bbox": det["bbox"], "age": 0}
                matched.add(best_id)
                det["track_id"] = best_id
            else:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {"bbox": det["bbox"], "age": 0}
                det["track_id"] = new_id

            tracked.append(det)

        # Age and remove old tracks
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["age"] += 1
            if self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]

        return tracked


def detect_on_crossing(
    frame: np.ndarray,
    vehicle_bbox: Tuple[int, int, int, int],
    plate_detector: Optional[PlateDetector] = None
) -> Optional[Dict[str, Any]]:
    """
    Detect license plate in vehicle bounding box region.
    Triggered when a vehicle crosses the line.

    Args:
        frame: Full video frame
        vehicle_bbox: (x1, y1, x2, y2) of vehicle
        plate_detector: PlateDetector instance

    Returns:
        Plate detection dict or None
    """
    if plate_detector is None:
        plate_detector = PlateDetector()

    # Detect in vehicle ROI
    detections = plate_detector.detect_in_roi(frame, vehicle_bbox)

    return detections[0] if detections else None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect license plates in images")
    parser.add_argument("image", nargs="?", help="Path to image file")
    parser.add_argument("--model", "-m", help="Path to plate detection model")
    parser.add_argument("--confidence", "-c", type=float, default=0.3,
                       help="Confidence threshold")
    parser.add_argument("--show", action="store_true", help="Display results")
    args = parser.parse_args()

    detector = PlateDetector(args.model, args.confidence)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Cannot read image: {args.image}")
            exit(1)

        detections = detector.detect(frame)

        print(f"Found {len(detections)} plate(s)")
        for i, det in enumerate(detections):
            print(f"  Plate {i+1}: {det['bbox']} (conf: {det['confidence']:.2f})")

        output = detector.draw_detections(frame, detections)
        cv2.imwrite("output/plate_detected.jpg", output)
        print("Saved: output/plate_detected.jpg")

        if args.show:
            cv2.imshow("Plate Detection", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    else:
        parser.print_help()
        print("\nUsage:")
        print("  python plate_detector.py data/images/traffic1/frame.jpg")
        print("  python plate_detector.py --model models/plate_detector.pt image.jpg")