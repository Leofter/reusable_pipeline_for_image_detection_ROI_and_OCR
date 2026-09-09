from abc import ABC, abstractmethod, abstractclassmethod
from numpy.typing import NDArray
import numpy as np
from ultralytics import YOLO

BoundingBoxes = NDArray[np.float32]


class Detection(ABC):

    @abstractmethod
    def predict(self, image_path: str, conf: float) -> BoundingBoxes:
        pass


class YoloDetection(Detection):

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict(self, image_path: str, conf: float) -> BoundingBoxes:
        results = self.model.predict(
            source=image_path, 
            conf=conf,
        )

        boxes = []
        for result in results:
            boxes.extend(result.boxes.xyxy.cpu().numpy())

        return boxes


def apply_detection(detection: Detection, image: str, conf: float):
    return detection.predict(image, conf)
