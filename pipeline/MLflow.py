import gc
import os
import re
from pathlib import Path

import mlflow
import torch
from ultralytics import YOLO, settings

DEFAULT_TRACKING_URI = f"sqlite:///{Path('mlflow.db').resolve()}"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "POC_yolo_models_for_boudingbox_detection"
)


def _sanitize_metric_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_./: -]", "_", name)


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


def yolo_model_val():

    settings.reset()
    settings.update({"mlflow": False})

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    device = 0 if torch.cuda.is_available() else "cpu"
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], filter_string="status = 'FINISHED'"
    )

    runs_df = runs_df[runs_df["tags.mlflow.runName"].str.endswith("-train", na=False)]

    for run_id in runs_df["run_id"]:

        gc.collect()
        torch.cuda.empty_cache()

        model_uri = f"runs:/{run_id}/weights/best.pt"

        try:
            local_model_path = mlflow.artifacts.download_artifacts(model_uri)
            model = YOLO(local_model_path)
        except Exception as error:
            raise RuntimeError(f"ERRO ao carregar modelo:{run_id} no YOLO()") from error

        with mlflow.start_run(run_id=run_id):
            print(f"\n\n\nAvaliando o modelo da Run ID: {run_id}\n\n\n")

            results = model.val(
                data="ignore/yaml/data.yaml",
                split="test",
                device=device,
                project="ignore/eval_models",
                name=f"eval_{run_id}",
            )

            metrics = {
                _sanitize_metric_name(f"test/{key}"): float(value)
                for key, value in results.results_dict.items()
            }

            metrics.update(
                {
                    f"test/speed_{process_name}_ms": float(results.speed[process_name])
                    for process_name in ("preprocess", "inference", "postprocess")
                    if process_name in results.speed
                }
            )

            mlflow.log_metrics(metrics)

            eval_dir = Path("ignore/eval_models") / f"eval_{run_id}"

            if eval_dir.exists():
                mlflow.log_artifacts(str(eval_dir), artifact_path="test")

    print(f"Concluído: {run_id}")
    del results
    del model
