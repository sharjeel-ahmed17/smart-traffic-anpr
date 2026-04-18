import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from functools import partial

# Patch torch.load to use weights_only=False for PyTorch 2.6+ compatibility
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

# Color map for each vehicle class (BGR format for OpenCV)
CLASS_COLORS = {
    2: (0, 255, 0),      # Car - Green
    3: (0, 0, 255),      # Motorbike - Red
    5: (255, 0, 0),      # Bus - Blue
    7: (255, 255, 0)     # Truck - Cyan
}


class VehicleDetector:
    """
    YOLOv8-based vehicle detector using COCO pretrained model.
    Detects: car, motorbike, bus, truck
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the vehicle detector.

        Args:
            model_path: Path to custom YOLOv8 model. If None, uses pretrained COCO model.
        """
        if model_path:
            self.model = YOLO(model_path)
        else:
            self.model = YOLO("yolov8n.pt")  # Nano model for speed

        # Set model to use GPU if available
        # self.model.to("cuda")  # Uncomment for GPU

    def detect(self, frame: np.ndarray, confidence: float = 0.3) -> List[Dict[str, Any]]:
        """
        Detect vehicles in a single frame.

        Args:
            frame: BGR image frame from OpenCV
            confidence: Minimum confidence threshold

        Returns:
            List of detection dictionaries with keys:
                - class_id: COCO class ID
                - class_name: Vehicle class name
                - confidence: Detection confidence
                - bbox: [x1, y1, x2, y2] bounding box
                - center: (cx, cy) center point of bounding box
        """
        results = self.model(frame, conf=confidence, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                box = boxes[i]
                cls_id = int(box.cls[0])

                # Filter only vehicle classes
                if cls_id not in VEHICLE_CLASSES:
                    continue

                # Get bounding box coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)

                # Calculate center point
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                detections.append({
                    "class_id": cls_id,
                    "class_name": VEHICLE_CLASSES[cls_id],
                    "confidence": float(box.conf[0]),
                    "bbox": [x1, y1, x2, y2],
                    "center": (cx, cy)
                })

        return detections

    def detect_batch(self, frames: List[np.ndarray],
                     confidence: float = 0.3) -> List[List[Dict[str, Any]]]:
        """
        Detect vehicles in a batch of frames.

        Args:
            frames: List of BGR image frames
            confidence: Minimum confidence threshold

        Returns:
            List of detection lists (one per frame)
        """
        results = self.model(frames, conf=confidence, verbose=False)
        all_detections = []

        for result in results:
            frame_detections = []
            boxes = result.boxes

            if boxes is not None:
                for i in range(len(boxes)):
                    box = boxes[i]
                    cls_id = int(box.cls[0])

                    if cls_id not in VEHICLE_CLASSES:
                        continue

                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    frame_detections.append({
                        "class_id": cls_id,
                        "class_name": VEHICLE_CLASSES[cls_id],
                        "confidence": float(box.conf[0]),
                        "bbox": [x1, y1, x2, y2],
                        "center": (cx, cy)
                    })

            all_detections.append(frame_detections)

        return all_detections

    def draw_detections(self, frame: np.ndarray,
                        detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw detection bounding boxes on frame.

        Args:
            frame: BGR image frame
            detections: List of detection dictionaries

        Returns:
            Frame with drawn bounding boxes
        """
        output = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = CLASS_COLORS.get(det["class_id"], (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(output, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return output

    def count_by_class(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Count detected vehicles by class.

        Args:
            detections: List of detection dictionaries

        Returns:
            Dictionary mapping class names to counts
        """
        counts = {}
        for det in detections:
            name = det["class_name"]
            counts[name] = counts.get(name, 0) + 1
        return counts


def detect_vehicles(frame: np.ndarray,
                    detector: Optional[VehicleDetector] = None,
                    confidence: float = 0.3) -> List[Dict[str, Any]]:
    """
    Convenience function to detect vehicles in a frame.

    Args:
        frame: BGR image frame
        detector: VehicleDetector instance (creates one if None)
        confidence: Confidence threshold

    Returns:
        List of detection dictionaries
    """
    if detector is None:
        detector = VehicleDetector()
    return detector.detect(frame, confidence)


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Detect vehicles in images using YOLOv8")
    parser.add_argument("image", nargs="?", help="Path to image file")
    parser.add_argument("--dir", help="Directory containing images")
    parser.add_argument("--output", "-o", default="output/detections",
                        help="Output directory for results")
    parser.add_argument("--confidence", "-c", type=float, default=0.3,
                        help="Confidence threshold")
    parser.add_argument("--show", action="store_true",
                        help="Display results with OpenCV window")
    args = parser.parse_args()

    detector = VehicleDetector()
    os.makedirs(args.output, exist_ok=True)

    if args.image:
        # Process single image
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Error: Cannot read image {args.image}")
            exit(1)

        detections = detector.detect(frame, args.confidence)
        output = detector.draw_detections(frame, detections)

        # Print results
        counts = detector.count_by_class(detections)
        print(f"\nDetections in {args.image}:")
        for cls, count in counts.items():
            print(f"  {cls}: {count}")
        print(f"  Total: {len(detections)}")

        # Save output
        basename = os.path.basename(args.image)
        output_path = os.path.join(args.output, f"detected_{basename}")
        cv2.imwrite(output_path, output)
        print(f"\nSaved: {output_path}")

        if args.show:
            cv2.imshow("Vehicle Detection", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif args.dir:
        # Process directory of images
        for filename in os.listdir(args.dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                filepath = os.path.join(args.dir, filename)
                frame = cv2.imread(filepath)

                if frame is not None:
                    detections = detector.detect(frame, args.confidence)
                    output = detector.draw_detections(frame, detections)

                    counts = detector.count_by_class(detections)
                    print(f"\n{filename}: {len(detections)} vehicles detected")
                    for cls, count in counts.items():
                        print(f"  {cls}: {count}")

                    output_path = os.path.join(args.output, f"detected_{filename}")
                    cv2.imwrite(output_path, output)

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python vehicle_detector.py data/images/traffic1/traffic1_frame_0000.jpg")
        print("  python vehicle_detector.py --dir data/images/traffic1 --show")
