import numpy as np

def iou(a, b):
    # a,b: [x1,y1,x2,y2]
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    w, h = max(0, xx2-xx1), max(0, yy2-yy1)
    inter = w*h
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    union = area_a + area_b - inter + 1e-6
    return inter/union

class SimpleTracker:
    def __init__(self, iou_thresh=0.5, max_age=15):
        self.tracks = {}     # id -> {box, age}
        self.next_id = 1
        self.iou_thresh = iou_thresh
        self.max_age = max_age

    def update(self, detections):
        # detections: [{"box":[...], "conf":.., "cls":..}, ...]
        assigned = set()
        updates = {}
        # match existing tracks
        for tid, t in list(self.tracks.items()):
            best = None
            best_iou = 0.0
            for j, det in enumerate(detections):
                if j in assigned: continue
                i = iou(t["box"], det["box"])
                if i > best_iou:
                    best_iou, best = i, j
            if best is not None and best_iou >= self.iou_thresh:
                det = detections[best]
                updates[tid] = {"box": det["box"], "age": 0}
                assigned.add(best)
            else:
                t["age"] += 1
                if t["age"] <= self.max_age:
                    updates[tid] = t  # keep alive
        # new tracks for unassigned dets
        for j, det in enumerate(detections):
            if j in assigned: continue
            updates[self.next_id] = {"box": det["box"], "age": 0}
            self.next_id += 1
        self.tracks = updates
        # return list with track_id
        out = []
        for tid, t in self.tracks.items():
            out.append({"id": tid, "box": t["box"]})
        return out
