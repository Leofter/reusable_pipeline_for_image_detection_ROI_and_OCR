from abc import ABC, abstractmethod
import cv2
import numpy as np


class Roi(ABC):

    @abstractmethod
    def crop(self):
        pass


class Crop(Roi):

    def __init__(self, image):
        self.image = image

    def crop(self, detections):
        roi_images = []

        for image_path, boxes in detections:
            image = cv2.imread(image_path)

            for box in boxes:
                xmin, ymin, xmax, ymax = np.asarray(box, dtype=int)
                roi_images.append(image[ymin:ymax, xmin:xmax])

        return roi_images


def apply_roi(roi: Roi, xyxy):
    return roi.crop(xyxy)
