# scripts/extract_frames.py
import argparse
from pathlib import Path
import cv2

from Football_Play_Project.io.video_io import iterate_frames
from Football_Play_Project.io.annotations import save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--outdir", default="data/images/extracted")
    ap.add_argument("--fps", type=int, default=5)
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    index = []
    for i, frame in iterate_frames(args.video, fps_sample=args.fps):
        outp = Path(args.outdir) / f"frame_{i:06d}.jpg"
        cv2.imwrite(str(outp), frame)
        index.append({"idx": i, "path": str(outp)})
    save_json(index, Path(args.outdir) / "index.json")

if __name__ == "__main__":
    main()
