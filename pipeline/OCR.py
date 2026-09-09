from abc import ABC, abstractmethod, abstractclassmethod
import cv2
from paddleocr import TextRecognition


class OcrInference(ABC):

    def __init__(self, image, output_dir: str):
        self.image = image
        self.output_dir = output_dir

    @abstractmethod
    def ocr_exec(self):
        pass


class PaddleOCR(OcrInference):

    def __init__(self, model, image, output_dir):
        super().__init__(image, output_dir)
        self.model = model

    def ocr_exec(self):
        model = TextRecognition(model_name=self.model)
        output = model.predict(input=self.image, batch_size=1)

        for res in output:
            res.print()
            res.save_to_json(save_path=self.output_dir + "/res.json")

        return output


def apply_ocr(ocr: OcrInference, model: str, image: str, output_dir: str) -> None:
    return ocr(model, image, output_dir).ocr_exec()
