# AGENTS.md

## Smart Cities: AI-Based Vehicle Tracking and License Plate Recognition System

### Problem Statement

Modern smart cities face increasing traffic congestion, law violations, and inefficient monitoring systems. Manual surveillance is costly, slow, and error-prone.

This project builds an AI-powered automated traffic monitoring system that can detect vehicles, count them, and recognize license plates in real-time for better traffic management and law enforcement.

### Objectives

- Detect vehicles (car, bike, truck, bus) in real-time video streams
- Count vehicles crossing a virtual line
- Extract license plate numbers using OCR when vehicles cross the virtual line
- Store vehicle type, license plate, and timestamp in a database
- Provide analytics for smart city traffic management

### System Architecture

```
Video Input
    ↓
Frame Extraction (OpenCV)
    ↓
YOLOv8 Vehicle Detection Model (pretrained on COCO dataset)
    ↓
DeepSORT / ByteTrack Tracking
    ↓
Line Crossing Detection Module
    ↓
License Plate Detection Model (YOLO ANPR)
    ↓
OCR Engine (EasyOCR / Tesseract)
    ↓
Database Storage (SQLite / PostgreSQL)
    ↓
Streamlit Dashboard / FastAPI Service
```

### Step-by-Step Workflow

1. **Data Input**
   - Load CCTV footage or traffic dataset videos
   - Extract frames using OpenCV

2. **Vehicle Detection**
   - Use pretrained YOLOv8 model (COCO dataset)
   - Detect: Car, Bike, Truck, Bus

3. **Tracking**
   - Apply DeepSORT or ByteTrack
   - Assign unique ID to each vehicle
   - Ensure same vehicle is not counted multiple times

4. **Line Crossing**
   - Define a virtual line on road
   - Detect centroid crossing event
   - Increment vehicle counter per class

5. **License Plate Detection**
   - Apply YOLO-based license plate detection model
   - Trigger only when vehicle crosses line

6. **OCR**
   - Crop license plate region
   - Apply EasyOCR / Tesseract
   - Convert image → text (plate number)

7. **Database (SQLite)**
   - Store structured records:
     - Vehicle type
     - License plate number
     - Timestamp of event

8. **Dashboard**
   - Live vehicle count (class-wise)
   - License plate logs
   - Traffic analytics graphs
   - Historical reports

### Expected Output

- Vehicle detection and counting
- License plate recognition
- Traffic analytics dashboard
- Stored logs in database

**Example Database Output:**

| Vehicle Type | Plate Number | Timestamp |
|--------------|--------------|-----------|
| Truck        | ABC-123      | 10:01 AM  |
| Truck        | XYZ-456      | 10:02 AM  |
| Truck        | DEF-789      | 10:05 AM  |

### Tech Stack

- **Language**: Python
- **Computer Vision**: OpenCV, YOLOv8
- **Tracking**: DeepSORT, ByteTrack
- **OCR**: EasyOCR, Tesseract
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Database**: SQLite / PostgreSQL

### Datasets

- **License Plate Dataset**: https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e
- **Sample Videos**: https://drive.google.com/drive/folders/11n5u1B3BAppISJl7OgLNII3shY9FnLsV?usp=sharing

### Impact

This system helps smart cities automate traffic monitoring, reduce manual surveillance, and improve law enforcement efficiency.
