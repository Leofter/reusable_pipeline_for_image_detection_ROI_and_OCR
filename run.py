import cv2
from ultralytics import YOLO
from paddleocr import TextRecognition
import os
from dotenv import load_dotenv


from pipeline import ROI as roi
from pipeline import OCR as ocr
from pipeline import detection as dt

load_dotenv()
# CONFIG YOLO
yolo_model = os.getenv("YOLO_MODEL")
image_path = os.getenv("IMAGE_PATH")
conf = 0.5


# CONFIG OCR
ocr_model = "PP-OCRv6_medium_rec"
ocr_output = "ignore/ocr_output"

#init Detection and OCR
detector = dt.YoloDetection(yolo_model)
ocr_init = ocr.PaddleOCR(ocr_model)

# RUN
detection_result = dt.apply_detection(detector, image_path, conf)

image_roi = roi.apply_roi(roi.Crop, image_path, detection_result)

ocr_result = ocr.apply_ocr(ocr_init, image_roi, ocr_output)
