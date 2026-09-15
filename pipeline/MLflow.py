import os
import mlflow
from pathlib import Path
from ultralytics import settings, YOLO
import gc
import torch

DEFAULT_TRACKING_URI = f"sqlite:///{Path('mlflow.db').resolve()}"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "POC_yolo_models_for_boudingbox_detection"
)


def reset_yolo_settings():
    settings.reset()


def set_mlflow_yolo_settings():
    settings.reset()
    settings.update({"mlflow": True})


def yolo_model_traning():

    settings.update({"mlflow": True})

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    os.environ["MLFLOW_TRACKING_URI"] = TRACKING_URI
    os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT_NAME

    models_dir = Path("ignore/yolo_models")

    for model in models_dir.iterdir():

        gc.collect()
        torch.cuda.empty_cache()

        if not model.is_file() and model.suffix != ".pt":
            raise ValueError("Nao tem apenas modelos dentro da pasta")
        else:
            model_name = model.name
            ymodel = YOLO(model)

            os.environ["MLFLOW_RUN"] = f"{model_name}-train"

            with mlflow.start_run(run_name=f"{model_name}-train"):
                results = ymodel.train(
                    data="ignore/yaml/data.yaml",
                    epochs=100,
                    batch=8,
                    patience=20,
                    imgsz=640,
                    cache="disk",
                    workers=2,
                    device=0,
                    project="ignore/train_models",
                    name=model_name,
                )

                train_dir = Path("ignore/train_models") / model_name
                if train_dir.exists():
                    mlflow.log_artifacts(str(train_dir), artifact_path="training")

        print(f"Concluido o treino do modelo: {model.name}")
        del results
        del ymodel


def yolo_model_validation():

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    models_dir = Path("")

    for model in models_dir.iterdir():
        if model.is_file() == False and model.suffix() != ".pt":
            raise ValueError("Nao tem apenas modelos dentro da pasta")
        else:
            model_name = model.name
            ymodel = YOLO(model)
            with mlflow.start_run(run_name="exp1-validation"):
                metrics = model.val(data="ignore/yaml/data.yaml", split="val")
                mlflow.log_metrics(
                    {
                        "val_mAP50": metrics.box.map50,
                        "val_mAP50_95": metrics.box.map,
                        "val_precision": metrics.box.mp,
                        "val_recall": metrics.box.mr,
                    }
                )
                # sobe a matriz de confusão, curva PR, curva F1, etc. geradas pelo val()
                mlflow.log_artifacts(str(metrics.save_dir), artifact_path="validation")
                mlflow.set_tags(
                    {
                        "source_train_run": "exp1",
                        "stage": "validation",
                    }
                )
