from abc import ABC, abstractmethod, abstractclassmethod
import cv2
from paddleocr import TextRecognition


class OcrInference(ABC):

    @abstractmethod
    def predict(self, image, output_dir: str):
        pass


class PaddleOCR(OcrInference):

    def __init__(self, model):
        self.model = TextRecognition(model_name=model)

    def predict(self, image, output_dir: str):
        output = self.model.predict(input=image, batch_size=1)

        for res in output:
            res.print()
            res.save_to_json(save_path=output_dir + "/res.json")

        return output


def apply_ocr(ocr: OcrInference, image, output_dir: str):
    return ocr.predict(image, output_dir)
