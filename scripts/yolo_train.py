# scripts/yolo_train.py
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO

def load_cfg(path: str | Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser(description="Train YOLO with Ultralytics")
    ap.add_argument("--data", default="src/Football_Play_Project/config/dataset.yaml",
                    help="Path to YOLO dataset yaml")
    ap.add_argument("--cfg", default="src/Football_Play_Project/config/model_yolo.yaml",
                    help="Path to model/training config yaml")
    ap.add_argument("--model", default=None, help="Override model weights (e.g., yolov8n.pt or runs/exp/weights/best.pt)")
    ap.add_argument("--imgsz", type=int, default=None, help="Override image size")
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs")
    ap.add_argument("--batch", type=int, default=None, help="Override batch size")
    ap.add_argument("--device", default=None, help="cuda, 0, 0,1, or cpu")
    ap.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    ap.add_argument("--project", default="runs/yolo", help="Ultralytics project dir")
    ap.add_argument("--name", default="train", help="Run name under project")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)

    # Allow CLI overrides
    if args.model:   cfg["model"] = args.model
    if args.imgsz:   cfg["imgsz"] = args.imgsz
    if args.epochs:  cfg["epochs"] = args.epochs
    if args.batch:   cfg["batch"]  = args.batch

    model_path = cfg.get("model", "yolov8n.pt")
    model = YOLO(model_path)

    # Pull common keys with sane defaults
    imgsz    = cfg.get("imgsz", 1280)
    epochs   = cfg.get("epochs", 50)
    batch    = cfg.get("batch", 16)
    patience = cfg.get("patience", 20)
    optimizer= cfg.get("optimizer", "auto")
    close_mosaic = cfg.get("close_mosaic", 10)
    lr0      = cfg.get("lr0", 0.01)
    lrf      = cfg.get("lrf", 0.01)

    results = model.train(
        data=args.data,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        patience=patience,
        optimizer=optimizer,
        close_mosaic=close_mosaic,
        lr0=lr0,
        lrf=lrf,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
    )
    print("\n✅ Training complete.")
    print(f"   Project: {results.get('project')}")
    print(f"   Name:    {results.get('name')}")
    print(f"   Weights: {results.get('save_dir')}/weights/best.pt")

if __name__ == "__main__":
    main()
