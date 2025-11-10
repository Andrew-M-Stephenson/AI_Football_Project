from ultralytics import YOLO
import numpy as np

class YoloRunner:
    def __init__(self, model_path: str, imgsz: int = 1280, conf: float = 0.25,
                 allowed_classes=None):
        """
        allowed_classes: list of class indices to keep (e.g., [0] for 'person' in COCO).
        If None, keep all classes.
        """
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.allowed_classes = allowed_classes

    def infer_frame(self, frame_bgr: np.ndarray):
        res = self.model.predict(
            source=frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=False
        )[0]

        out = []
        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        clses = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for b, c, p in zip(xyxy, clses, confs):
            c = int(c)
            if self.allowed_classes is not None and c not in self.allowed_classes:
                continue
            out.append({
                "box": b.tolist(),
                "cls": c,
                "conf": float(p),
            })
        return out
