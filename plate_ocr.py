"""
License Plate OCR - Extract text from license plate images using EasyOCR.
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache


# Try to import EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Try to import Tesseract
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class PlateOCR:
    """
    OCR for license plate text extraction.
    Uses EasyOCR by default, falls back to Tesseract or simple methods.
    """

    def __init__(self, language: str = 'en', use_gpu: bool = False,
                 reader_type: str = "easyocr"):
        """
        Initialize OCR.

        Args:
            language: Language code(s) for OCR (default: 'en')
            use_gpu: Use GPU for EasyOCR
            reader_type: 'easyocr', 'tesseract', or 'auto'
        """
        self.language = language
        self.use_gpu = use_gpu
        self.reader_type = reader_type
        self._reader = None
        self._init_reader()

    def _init_reader(self):
        """Initialize the OCR reader."""
        if self.reader_type == "easyocr" and EASYOCR_AVAILABLE:
            try:
                # Try to load EasyOCR reader
                lang_list = [self.language] if isinstance(self.language, str) else self.language
                self._reader = easyocr.Reader(lang_list, gpu=self.use_gpu, verbose=False)
                print(f"EasyOCR initialized with language: {self.language}")
            except Exception as e:
                print(f"Failed to initialize EasyOCR: {e}")
                self._reader = None

        elif self.reader_type == "tesseract" and TESSERACT_AVAILABLE:
            self._reader = pytesseract
            print("Tesseract OCR initialized")

        else:
            # Auto-select
            if EASYOCR_AVAILABLE:
                self.reader_type = "easyocr"
                self._init_reader()
            elif TESSERACT_AVAILABLE:
                self.reader_type = "tesseract"
                self._init_reader()
            else:
                print("Warning: No OCR backend available. Using simple method.")

    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.

        Args:
            image: Input plate image

        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply histogram equalization for better contrast
        equalized = cv2.equalizeHist(gray)

        # Apply slight blur to reduce noise
        blurred = cv2.GaussianBlur(equalized, (3, 3), 0)

        # Apply thresholding
        _, thresh = cv2.threshold(blurred, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh

    def read_text_easyocr(self, image: np.ndarray,
                         detail: bool = False) -> List[Dict[str, Any]]:
        """Read text using EasyOCR."""
        if self._reader is None:
            return []

        try:
            # Convert BGR to RGB for EasyOCR
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            results = self._reader.readtext(image_rgb)

            detections = []
            for (bbox, text, confidence) in results:
                # Filter low confidence results
                if confidence < 0.3:
                    continue

                # Clean text
                cleaned = text.strip().upper()
                if cleaned:
                    detections.append({
                        "text": cleaned,
                        "confidence": float(confidence),
                        "bbox": bbox
                    })

            return detections

        except Exception as e:
            print(f"EasyOCR error: {e}")
            return []

    def read_text_tesseract(self, image: np.ndarray,
                          config: str = "--psm 7") -> List[Dict[str, Any]]:
        """Read text using Tesseract."""
        if self._reader is None:
            return []

        try:
            # Preprocess
            processed = self.preprocess_for_ocr(image)

            # Run Tesseract
            text = self._reader.image_to_string(processed, config=config)

            # Parse results
            lines = text.strip().split('\n')
            detections = []

            for line in lines:
                cleaned = line.strip()
                if cleaned and len(cleaned) >= 2:
                    # Estimate confidence (Tesseract doesn't give per-char confidence easily)
                    detections.append({
                        "text": cleaned.upper(),
                        "confidence": 0.8,  # Default
                        "bbox": None
                    })

            return detections

        except Exception as e:
            print(f"Tesseract error: {e}")
            return []

    def read_text(self, image: np.ndarray,
                 detail: bool = False) -> List[Dict[str, Any]]:
        """
        Read text from plate image.

        Args:
            image: Plate image (BGR or grayscale)
            detail: Return detailed results with bounding boxes

        Returns:
            List of text detections with confidence
        """
        if self.reader_type == "easyocr" and self._reader:
            return self.read_text_easyocr(image, detail)
        elif self.reader_type == "tesseract" and self._reader:
            return self.read_text_tesseract(image)
        else:
            # Simple fallback
            return self._simple_ocr(image)

    def _simple_ocr(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Simple OCR fallback using contour detection."""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Threshold
            _, thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Find contours (potential character regions)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return []

            # Get image dimensions
            h, w = gray.shape

            # Filter by size (characters should be reasonable size)
            valid_contours = []
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if 0.1 < cw / w < 0.9 and 0.2 < ch / h < 0.9:
                    valid_contours.append((x, y, cw, ch))

            if not valid_contours:
                return []

            # Sort left to right
            valid_contours.sort(key=lambda x: x[0])

            # Return placeholder (simple method can't actually read text)
            return [{
                "text": "[plate]",
                "confidence": 0.3,
                "bbox": None
            }]

        except Exception as e:
            print(f"Simple OCR error: {e}")
            return []

    def read_plate_number(self, image: np.ndarray) -> Optional[str]:
        """
        Read the primary plate number from an image.

        Args:
            image: Plate image

        Returns:
            Extracted plate number or None
        """
        results = self.read_text(image)

        if not results:
            return None

        # Return the highest confidence result
        # Sort by confidence
        results.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        return results[0].get('text', None)

    def read_from_file(self, image_path: str) -> Optional[str]:
        """
        Read plate number from image file.

        Args:
            image_path: Path to plate image

        Returns:
            Extracted plate number or None
        """
        image = cv2.imread(image_path)

        if image is None:
            print(f"Cannot read image: {image_path}")
            return None

        return self.read_plate_number(image)

    def batch_read(self, image_paths: List[str]) -> Dict[str, str]:
        """
        Read plate numbers from multiple files.

        Args:
            image_paths: List of image paths

        Returns:
            Dictionary mapping filepath to plate number
        """
        results = {}

        for path in image_paths:
            plate = self.read_from_file(path)
            if plate:
                results[path] = plate
            else:
                results[path] = "[unreadable]"

        return results


def read_plate(image_path: str, reader_type: str = "easyocr") -> Optional[str]:
    """
    Convenience function to read plate number.

    Args:
        image_path: Path to plate image
        reader_type: OCR reader type

    Returns:
        Extracted plate number
    """
    ocr = PlateOCR(reader_type=reader_type)
    return ocr.read_from_file(image_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="License Plate OCR")
    parser.add_argument("image", help="Path to plate image")
    parser.add_argument("--type", choices=["easyocr", "tesseract", "auto"],
                       default="auto", help="OCR type")
    parser.add_argument("--lang", default="en", help="Language code")
    args = parser.parse_args()

    ocr = PlateOCR(language=args.lang, reader_type=args.type)

    print(f"\nReading plate from: {args.image}")
    result = ocr.read_from_file(args.image)

    if result:
        print(f"  Plate number: {result}")
    else:
        print("  Could not read plate")