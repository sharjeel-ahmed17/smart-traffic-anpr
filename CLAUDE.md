# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Smart City Traffic ANPR (Automatic Number Plate Recognition) System - an AI-powered traffic monitoring solution that detects vehicles, tracks them across video frames, counts line crossings, and performs license plate OCR.

## System Architecture

```
Video Input → Frame Extraction (OpenCV) → YOLOv8 Detection → DeepSORT/ByteTrack Tracking 
              → Line Crossing Detection → License Plate Detection → OCR (EasyOCR) 
              → Database (SQLite) → Dashboard (Streamlit/FastAPI)
```

## Directory Structure

- `app/` - Streamlit dashboard application
- `models/` - Pretrained YOLO models (`plate_model.pt`, `vehicle_model.pt`)
- `database/` - SQLite database (`traffic.db`) and schema (`schema.sql`)
- `data/` - Input videos (`data/videos/`) and images (`data/images/`)
- `notebooks/` - Development notebooks for vehicle detection, tracking, and license plate OCR
- `output/` - Processed output files

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit dashboard
streamlit run app/streamlit_app.py

# Run FastAPI server (if implemented)
uvicorn app.main:app --reload
```

## Key Technologies

- **Detection**: YOLOv8 (Ultralytics) for vehicle and license plate detection
- **Tracking**: DeepSORT/ByteTrack for vehicle ID assignment
- **OCR**: EasyOCR for license plate text extraction
- **Backend**: FastAPI for API endpoints
- **Frontend**: Streamlit for real-time dashboard
- **Database**: SQLite for storing vehicle logs (type, plate, timestamp)

## Development Notes

- Virtual line crossing triggers license plate capture to avoid processing every frame
- DeepSORT ensures vehicles aren't double-counted
- Models are stored in `models/` directory (vehicle detection and plate detection)
