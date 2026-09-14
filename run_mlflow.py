import mlflow
from ultralytics import settings, YOLO
from pipeline import MLflow as ml

# TROCAR QUANDO FOR FAZER O SOLID
# os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
# os.environ["MLFLOW_EXPERIMENT_NAME"] = "POC_yolo_models_for_boudingbox_detection"
# os.environ["MLFLOW_RUN"] = "exp1-train"

ml.yolo_model_traning()
print("Concluido")
