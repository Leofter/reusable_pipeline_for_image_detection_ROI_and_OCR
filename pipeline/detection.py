from abc import ABC, abstractmethod, abstractclassmethod
from ultralytics import YOLO


class Detection(ABC):

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def model_result(self):
        pass


class YoloDetection(Detection):

    def __init__(self, model, image_path: str):
        super().__init__(model)
        self.image_path = image_path

    def model_result(self, conf: float):
        self.conf = conf

        model = YOLO(self.model)

        results = model.predict(
            source=self.image_path,  # Path to your test or validation images
            conf=self.conf,
        )

        for result in results:
            xyxy = result.boxes.xyxy
            # top-left-x, top-left-y, bottom-right-x, bottom-right-y

        return xyxy
