# scripts/compute_accuracy.py
import csv
import argparse
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="runs/cnn/prediction_log.csv")
    args = ap.parse_args()

    rows = []
    with open(args.log, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            #normalize fields
            row["correct"] = int(row.get("correct","0") or 0)
            row["final_pred"] = row.get("final_pred","").strip()
            row["true_label"] = row.get("true_label","").strip()
            rows.append(row)

    if not rows:
        print("No rows in log.")
        return

    correct = sum(r["correct"] for r in rows)
    print(f"Total samples: {len(rows)}")
    print(f"Overall accuracy: {correct/len(rows):.3f}")

    buckets = defaultdict(list)
    for r in rows:
        cls = r["true_label"] if r["true_label"] else r["final_pred"]
        if not cls:
            cls = "<unknown>"
        buckets[cls].append(r["correct"])

    print("\nPer-class accuracy:")
    for cls, vals in sorted(buckets.items()):
        acc = sum(vals) / max(1, len(vals))
        print(f"  {cls:>12s}: {acc:.3f}  (n={len(vals)})")

if __name__ == "__main__":
    main()
