from abc import ABC, abstractmethod, abstractclassmethod
import cv2
from .detection import *


class Roi(ABC):

    def __init__(self, image, xyxy):
        self.image = image
        self.xyxy = xyxy

    @abstractmethod
    def crop(self):
        pass


class Crop(Roi):

    def __init__(self, image, xyxy):
        super().__init__(image, xyxy)

    def crop(self):
        image = cv2.imread(self.image)
        results = xyxy

        for result in results:
            for box in result.boxes:

                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                xmin, ymin, xmax, ymax = xyxy

                self.roi_image = image[ymin:ymax, xmin:xmax]

                return self.roi_image


def apply_roi(roi: Roi, image: str, xyxy):
    return Roi.crop(image, xyxy)