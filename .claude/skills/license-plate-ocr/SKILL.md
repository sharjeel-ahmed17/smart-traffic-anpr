---
name: license-plate-ocr
description: >
  Detect license plates in vehicle images using a YOLO-based plate detection model, then extract
  the plate number using EasyOCR or Tesseract OCR. Use this skill whenever the user needs:
  ANPR (Automatic Number Plate Recognition), license plate reading, OCR on number plates,
  plate detection + text extraction, read plate number from image/frame, crop and OCR license plate,
  EasyOCR for plates, Tesseract ANPR. This skill is triggered by the line-crossing-counter skill
  and feeds results to the traffic-database skill.
  Dataset: https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e
---

# License Plate OCR Skill

Detect license plates in a vehicle crop using YOLO, then extract the plate number text with EasyOCR or Tesseract.

---

## Dependencies

```bash
pip install ultralytics easyocr opencv-python Pillow

# Optional: Tesseract backend
sudo apt install tesseract-ocr
pip install pytesseract
```

---

## Dataset & Model

**Training dataset (Roboflow):**

```
https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e
```

**Download pre-trained plate model** or train your own:

```python
from ultralytics import YOLO

# Train on the Roboflow dataset
model = YOLO("yolov8n.pt")
model.train(data="license-plate-dataset/data.yaml", epochs=50, imgsz=640)
model.save("models/plate_model.pt")
```

**Or load pre-trained:**

```python
plate_model = YOLO("models/plate_model.pt")
```

---

## Step 1: Detect License Plate in Vehicle Region

```python
import cv2
import numpy as np
from ultralytics import YOLO

def detect_plate_region(frame, vehicle_bbox, plate_model, conf=0.4):
    """
    Crop vehicle from frame and detect license plate within it.

    Args:
        frame: Full video frame (BGR numpy array)
        vehicle_bbox: (x1, y1, x2, y2) of vehicle bounding box
        plate_model: Loaded YOLO plate detection model

    Returns:
        Cropped plate image (numpy array) or None if not found
    """
    x1, y1, x2, y2 = vehicle_bbox

    # Add small padding
    pad = 10
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(frame.shape[1], x2 + pad)
    y2 = min(frame.shape[0], y2 + pad)

    vehicle_crop = frame[y1:y2, x1:x2]

    results = plate_model(vehicle_crop, conf=conf, verbose=False)[0]

    if len(results.boxes) == 0:
        return None

    # Take highest confidence detection
    best = max(results.boxes, key=lambda b: float(b.conf[0]))
    px1, py1, px2, py2 = map(int, best.xyxy[0])

    plate_crop = vehicle_crop[py1:py2, px1:px2]
    return plate_crop
```

---

## Step 2A: OCR with EasyOCR (Recommended)

```python
import easyocr

# Initialize once (slow first time — downloads models)
reader = easyocr.Reader(['en'], gpu=False)

def read_plate_easyocr(plate_image):
    """
    Extract text from a cropped license plate image.
    Returns cleaned plate string or None.
    """
    if plate_image is None or plate_image.size == 0:
        return None

    # Preprocess: upscale + grayscale
    plate_image = preprocess_plate(plate_image)

    results = reader.readtext(plate_image)

    if not results:
        return None

    # Concatenate all detected text segments
    texts = [r[1] for r in results if r[2] > 0.3]  # filter by confidence
    plate_text = "".join(texts).upper().strip()
    plate_text = clean_plate_text(plate_text)

    return plate_text if plate_text else None
```

---

## Step 2B: OCR with Tesseract (Alternative)

```python
import pytesseract
from PIL import Image

def read_plate_tesseract(plate_image):
    """
    Use Tesseract for plate OCR.
    """
    if plate_image is None or plate_image.size == 0:
        return None

    plate_image = preprocess_plate(plate_image)
    pil_img = Image.fromarray(plate_image)

    config = "--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
    text = pytesseract.image_to_string(pil_img, config=config)
    return clean_plate_text(text.strip().upper())
```

---

## Step 3: Image Preprocessing

Good preprocessing dramatically improves OCR accuracy:

```python
def preprocess_plate(plate_img):
    """
    Resize, grayscale, and threshold the license plate crop.
    """
    # Upscale to at least 200px wide
    h, w = plate_img.shape[:2]
    if w < 200:
        scale = 200 / w
        plate_img = cv2.resize(plate_img, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, h=10)

    return denoised
```

---

## Step 4: Clean Plate Text

```python
import re

def clean_plate_text(text):
    """
    Remove invalid characters from OCR output.
    Keeps only alphanumeric and dash.
    """
    cleaned = re.sub(r"[^A-Z0-9\-]", "", text.upper())
    return cleaned if len(cleaned) >= 2 else None
```

---

## Step 5: Combined Entry Point

```python
def detect_and_read_plate(frame, vehicle_bbox,
                          plate_model, ocr_engine="easyocr"):
    """
    Full pipeline: detect plate in vehicle crop → OCR.

    Returns plate number string or "UNKNOWN".
    """
    plate_crop = detect_plate_region(frame, vehicle_bbox, plate_model)

    if plate_crop is None:
        return "UNKNOWN"

    if ocr_engine == "easyocr":
        return read_plate_easyocr(plate_crop) or "UNKNOWN"
    else:
        return read_plate_tesseract(plate_crop) or "UNKNOWN"
```

---

## OCR Engine Comparison

| Feature        | EasyOCR          | Tesseract        |
| -------------- | ---------------- | ---------------- |
| Accuracy       | ✅ Higher        | Moderate         |
| Speed          | Moderate         | ⚡ Faster        |
| GPU support    | ✅ Yes           | No               |
| Multi-language | ✅ 80+ languages | Limited          |
| Setup          | pip only         | System install   |
| Best for       | Most cases       | Resource-limited |

---

## Notes

- Only trigger this when a vehicle **crosses the virtual line** (see line-crossing-counter skill).
- Store results with timestamp via the **traffic-database** skill.
- If plate is not visible or vehicle is far away, return `"UNKNOWN"` — don't raise exceptions.
- For full pipeline integration, see the **anpr-pipeline** skill.
