import cv2

class Roi:

    def __init__(self, detection):
        self.detection = detection

    def roi_result(self):
        image = cv2.imread(self.detection.image)
        results = self.detection.model_result()

        for result in results:
            for box in result.boxes:

                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                xmin, ymin, xmax, ymax = xyxy

                self.roi_image = image[ymin:ymax, xmin:xmax]

                return self.roi_image

    def show_roi(self):

        cv2.imshow("ROI image", self.roi_image)
        cv2.waitKey(0)