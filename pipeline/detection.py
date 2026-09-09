from abc import ABC, abstractmethod, abstractclassmethod
from ultralytics import YOLO


class Detection(ABC):

    @abstractmethod
    def predict(self, image_path: str, conf: float):
        pass


class YoloDetection(Detection):

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict(self, image_path: str, conf: float):
        results = self.model.predict(
            source=image_path,  # Path to your test or validation images
            conf=conf,
        )

        boxes = []
        for result in results:
            boxes.extend(result.boxes.xyxy.cpu().numpy())

        return boxes


def apply_detection(detection: Detection, image: str, conf: float):
    return detection.predict(image, conf)
