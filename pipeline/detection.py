from abc import ABC, abstractmethod, abstractclassmethod
from ultralytics import YOLO


class Detection(ABC):

    @abstractmethod
    def model_result(self, model: str, image_path: str, conf: float):
        pass


class YoloDetection(Detection):

    def model_result(self, model: str, image_path: str, conf: float):

        model = YOLO(model)

        results = model.predict(
            source=image_path,  # Path to your test or validation images
            conf=conf,
        )

        for result in results:
            xyxy = result.boxes.xyxy
            # top-left-x, top-left-y, bottom-right-x, bottom-right-y

        return xyxy


def apply_detection(detection: Detection, model: str, image: str, conf: float):
    return detection.model_result(model, image, conf)
