import cv2


class OcrInference:

    def __init__(self, model: str, roi_image, output_dir: str):
        self.model = model
        self.roi_image = roi_image
        self.output_dir = output_dir

    def ocr_exec(self):
        model = TextRecognition(model_name=self.model)
        output = model.predict(input=self.roi_image, batch_size=1)

        for res in output:
            res.print()
            res.save_to_json(save_path=self.output_dir + "/res.json")

        return output
