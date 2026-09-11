from abc import ABC, abstractmethod, abstractclassmethod
import cv2
from .detection import *


class Roi(ABC):

    @abstractmethod
    def crop(self):
        pass


class Crop(Roi):

    def __init__(self, image):
        self.image = image

    def crop(self, xyxy):

        while True:
            image = cv2.imread(self.image)
            self.xyxy = xyxy

            for box in self.xyxy:
                xyxy = box.astype(int)
                xmin, ymin, xmax, ymax = xyxy

                self.roi_image = image[ymin:ymax, xmin:xmax]

            return self.roi_image


def apply_roi(roi: Roi, xyxy):
    return roi.crop(xyxy)
