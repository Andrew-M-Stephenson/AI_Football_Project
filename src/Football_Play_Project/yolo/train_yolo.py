from ultralytics import YOLO

def train(dataset_yaml: str, model_cfg: dict):
    model = YOLO(model_cfg.get("model", "yolov8n.pt"))
    model.train(
        data=dataset_yaml,
        imgsz=model_cfg.get("imgsz", 1280),
        epochs=model_cfg.get("epochs", 50),
        batch=model_cfg.get("batch", 16),
        patience=model_cfg.get("patience", 20),
        optimizer=model_cfg.get("optimizer", "auto"),
        close_mosaic=model_cfg.get("close_mosaic", 10),
        lr0=model_cfg.get("lr0", 0.01),
        lrf=model_cfg.get("lrf", 0.01),
    )
    return model
