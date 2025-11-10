import os
import subprocess

print("Marking Line of Scrimmage...")
subprocess.run(["python", "scripts/mark_los_manual.py"])

print("Running YOLO and manual player labeling...")
subprocess.run(["python", "scripts/label_positions.py"])

print("Assigning formations...")
subprocess.run(["python", "scripts/label_formation.py"])

print("✅ Full pipeline complete.")