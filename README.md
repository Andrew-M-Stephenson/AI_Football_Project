## Activate .venv ##

.venv\Scripts\Activate.ps1

---------------------------------

Download yolov8n.pt

---------------------------------

1. Big-picture pipeline

Your project has three main components:

YOLO detector (Ultralytics Yolov8n)

Detects all people on each frame.

Geometry + manual labels

You clean up detections, draw the LOS, and label each player as QB / RB / WR / OL / DEF.

A meta builder combines all of that into a structured JSON.

CNN play predictor

Takes the pre-snap layout (player positions + roles) and predicts one of your 4 play outcome classes
(e.g. Deep_Pass, Short_Pass, Middle_Run, Outside_Run).

Plus:

A real-time script that runs YOLO + heuristics + CNN on a video, with you drawing the LOS on the fly.

Everything in your repo is either:

Helping build the dataset (YOLO + LOS + positions → meta),

Training/evaluating the CNN, or

Running the live demo.

2. scripts/ – what each file does
detect_players_yolo.py

Purpose: Run YOLO on all your labeled play images and save raw detections.

Rough behavior:

Walks data/images/Plays/ recursively.
Example layout:

data/images/Plays/
  Deep_Pass/img001.jpg
  Short_Pass/img010.jpg
  Middle_Run/img050.jpg
  ...


For each image, runs:

YOLO("yolov8n.pt").predict(...)


Filters to the person class (class 0), giving bounding boxes [x1, y1, x2, y2].

Saves a JSON mapping:

{
  "Deep_Pass/img001.jpg": [
    {"bbox": [x1, y1, x2, y2], "conf": 0.87, ...},
    ...
  ],
  "Short_Pass/img010.jpg": [...],
  ...
}


to data/meta/plays_detections.json.

You ran this first to get detections before cleaning & labeling.

visualize_detections.py

Purpose: Visual sanity-check of YOLO detections.

Loads plays_detections.json.

Opens each image under data/images/Plays/.

Draws:

Green boxes around each detection

Index numbers (0, 1, 2, …)

A text overlay: file name and number of boxes

Keyboard controls:

D / → : next image

A / ← : previous image

Q / Esc : quit

This is how you checked “are there really 60 people here?” and whether YOLO is doing something sensible before you start manual labeling.

edit_detections_click.py

Purpose: Clean up noisy YOLO detections by clicking boxes to keep/remove.

Loads plays_detections.json.

For each image, initializes keep flags (all True).

Mouse handler:

Left-click inside a box → toggles between:

green = keep

red = delete

When you move to the next/previous image, it:

Filters out boxes you marked red.

Writes the cleaned boxes back to JSON (plays_detections.json or the cleaned output you chose).

Result: same JSON structure, but now only good player boxes remain, no random fans/sideline boxes.

This is essential before you label roles, because you don’t want to waste time labeling boxes you’re just going to ignore.

mark_los_manual.py (images LOS labeling)

Purpose: Let you manually draw the Line of Scrimmage for each image.

Walks through all images in data/images/Plays.

For each image:

You left-click two points along the LOS.

It draws a line between those two points (usually slightly slanted).

Press S to save, R to reset, N to skip, Q/Esc to quit.

Saves a JSON like:

{
  "Deep_Pass/img001.jpg": {"p1": [x1, y1], "p2": [x2, y2]},
  "Short_Pass/img010.jpg": {"p1": [...], "p2": [...]},
  ...
}


in data/meta/plays_los.json.

This is how you give geometry context: which players are on/near the LOS vs in the backfield.

label_positions_manual.py

Purpose: Manually assign positions to each player box: QB, RB, WR, OL, DEF.

Uses the cleaned detections in plays_detections.json.

For each image:

Draws all boxes with indices.

Highlights a “current box”.

You move between boxes with A/D.

You press:

1 → label current box as QB

2 → RB

3 → WR

4 → OL

5 → DEF

S → save labels for that image and go to next

B → save and go back

Q/Esc → save all and quit

Saves a JSON:

{
  "Deep_Pass/img001.jpg": {
    "0": "QB",
    "1": "WR",
    "2": "WR",
    "3": "OL",
    "4": "DEF",
    ...
  },
  "Short_Pass/img010.jpg": {
    "0": "QB",
    "1": "RB",
    "2": "OL",
    ...
  }
}


in data/meta/plays_positions.json.

These are your ground-truth roles that the CNN uses as inputs (features).

edit_detections_with_positions.py (newer tool – you kept edit_detections_click instead)

If you kept / use this: it’s like edit_detections_click.py but also keeps plays_positions.json in sync while you delete boxes.

When you delete a box (turn red), the script:

Removes that box from plays_detections.json

Drops its role from plays_positions.json

Reindexes the remaining boxes and roles

You ended up using the simpler version (edit_detections_click) earlier, but this is the more “correct” one when you’re refining both detections and roles after-the-fact.

build_meta_with_los.py (your build_plays_meta_full script)

Purpose: Combine detections, LOS, and positions into a single canonical meta JSON, plus a labels CSV.

It pulls from:

plays_detections.json → cleaned boxes

plays_positions.json → your QB/RB/WR/OL/DEF labels

plays_los.json → LOS line

The folder name under data/images/Plays/ → play outcome label, e.g. Deep_Pass

It then produces:

data/meta/plays_meta_full.json – for each frame:

{
  "frames": [
    {
      "idx": 0,
      "fname": "Deep_Pass/img001.jpg",
      "los": { "p1": [x1, y1], "p2": [x2, y2] },
      "detections": [
        { "box": [x1, y1, x2, y2], "role_pos": "QB",  "side": "offense" },
        { "box": [ ... ],          "role_pos": "WR",  "side": "offense" },
        { "box": [ ... ],          "role_pos": "DEF", "side": "defense" },
        ...
      ],
      "play_label": "Deep_Pass"
    },
    ...
  ]
}


idx is the numeric ID used by the CNN.

side is derived from the role (DEF → defense, others → offense).

data/meta/play_labels.csv – a simple mapping:

key_frame_idx,play_label
0,Deep_Pass
1,Short_Pass
2,Middle_Run
...


This is exactly what train_cnn.py reads as the label file.

This script is the bridge between your labeling work and the neural network training.

train_cnn.py (in scripts/)

Purpose: Train your CNN to map pre-snap formation → play outcome.

Parses arguments:

--meta data/meta/plays_meta_full.json

--labels data/meta/play_labels.csv

--epochs, --batch_size, etc.

Instantiates:

dataset = PlayDataset(meta_path=args.meta, labels_csv=args.labels, max_players=11)


from Football_Play_Project.cnn.dataset.

Splits into train/val via random_split.

Creates PlayCNN from Football_Play_Project.cnn.model.

Typical PyTorch training loop:

Forward pass: logits = model(X)

Loss: CrossEntropyLoss

Backprop + optimizer step

Tracks the best validation accuracy.

Saves best model to:

runs/cnn/model.pt


containing:

model_state (weights)

label2idx, idx2label (class names)

feature_dim, max_players

Training log you saw:

~177 plays

4 classes

Best val_acc ≈ 0.40 → better than random (0.25), so the model is learning real structure.

predict_cnn.py

Purpose: Test the CNN on a single labeled frame from plays_meta_full.json.

Loads runs/cnn/model.pt.

Loads plays_meta_full.json.

Finds the frame with idx == --frame_idx.

Re-creates the same feature tensor as the dataset:

Only offense players

For each: cx_norm, cy_norm, side_lr, one-hot role (QB/RB/WR/OL/DEF)

Sort left→right

Pad/truncate to max_players x feature_dim

Runs:

logits = model(X)
probs = softmax(logits)


Prints:

Predicted class label

Probability for each class

You use this to double-check that the model is doing something plausible on your training-style data.

realtime_play_predict.py

Purpose: Run the full stack on a video: YOLO + LOS (manual) + role heuristics + CNN prediction, with pause.

Key pieces:

Video capture

Reads from --video (file path or camera index).

Pause / resume

P toggles paused state.

When paused, it keeps reusing the same frame so you can draw LOS.

LOS interaction

L clears current LOS.

You click two points on the frame to set a new LOS line.

LOS is stored as {"p1": [x,y], "p2": [x,y]}.

YOLO people detection

Each (paused or playing) frame runs YOLO with classes=[0] (person).

Detections become dets = [{"box": [x1,y1,x2,y2]}, ...].

Heuristic role assignment & features

build_features_from_dets(dets, los_line, frame.shape, max_players)

Rough steps:

Normalize player centers to [0,1] by dividing by width/height.

Sort by cx (left→right).

Estimate a “LOS row” via median cy.

Choose middle row players as OL (up to 5).

QB = player just behind OL and near OL center.

RB = player behind QB.

WR = players far left/right from OL center.

Everyone else = OL fallback.

Build [cx, cy, side_lr, one_hot(role)] feature vectors.

Pad/truncate to max_players and shape [1, max_players, feature_dim].

CNN prediction

Uses the same PlayCNN you trained.

If LOS + dets exist:

logits = model(X)

probs = softmax(logits)

Draws predicted label + probabilities on the frame.

UI

Overlays text:

PLAYING | Pred: Deep_Pass or PAUSED | Press L, then click 2 points...

Shows class probabilities beneath.

Q/Esc to quit.

This script is how you demonstrate your “almost real-time” system: you pause at pre-snap, draw LOS, and see what the model thinks.

yolo_train.py / yolo_predict.py

These are more generic YOLO helpers (less central now, but still useful).

yolo_train.py

Thin wrapper to call ultralytics.YOLO().train(...).

Uses your dataset.yaml (if you set one up) to train a customized detector instead of plain yolov8n.pt.

yolo_predict.py

General YOLO inference script (similar to detect_players_yolo.py but more generic).

Used for quick “run YOLO on a video and save an annotated output”.

Right now your core pipeline uses detect_players_yolo.py for dataset building, but yolo_predict.py is still handy for quick visualization.

3. src/Football_Play_Project/cnn/ – your neural network code

This is the library code for the CNN, used by both train_cnn.py and predict_cnn.py / realtime_play_predict.py.

dataset.py

Defines:

ROLE_ORDER = ["QB", "RB", "WR", "OL", "DEF"]
→ This fixed order is how you map role strings to the one-hot vector.

PlayDataset (subclass of torch.utils.data.Dataset):

Inputs:

meta_path → plays_meta_full.json

labels_csv → play_labels.csv

max_players → typically 11

What it does:

Loads meta["frames"] and a key_frame_idx → play_label mapping from the CSV.

For each frame (that has a label):

Reads fname, detections, los, play_label.

Builds feature vector per offensive player:

Compute cx_norm and cy_norm from the bounding box.

side_lr = 0 for left half, 1 for right half.

Role one-hot embedding from role_pos and ROLE_ORDER.

So each player = [cx, cy, side_lr, one_hot_role...] → length 3 + len(ROLE_ORDER).

Sorts players left→right by cx.

Creates a fixed-size array [max_players, feature_dim]:

Fills first n rows with features.

Fills remaining rows with zeros if fewer than max_players.

Converts play label string to an integer via label2idx, yielding y.

__getitem__(i) returns (X, y) where:

X is [max_players, feature_dim] tensor.

y is an integer class index.

Stores:

feature_dim

num_classes

label2idx, idx2label, max_players

This ensures training and prediction both see the same feature structure.

model.py

Defines the CNN architecture: PlayCNN.

Conceptually:

Treats the formation as a sequence of players (left to right).

Applies 1D convolutions across players to capture patterns like “two WRs left, one RB in backfield, QB under center”.

Typical structure (high-level):

Input: [batch, max_players, feature_dim].

Permute to [batch, feature_dim, max_players] (channels-first for Conv1d).

Pass through a stack of Conv1d + nonlinearity layers:

e.g. Conv1d(feature_dim → 32, kernel_size=3, padding=1)

then Conv1d(32 → 64, kernel_size=3, padding=1)

Global pooling over players dimension (e.g. max or average) → [batch, 64].

Feed into a small fully-connected head:

e.g. Linear(64 → 64), ReLU, Dropout, Linear(64 → num_classes).

Output: [batch, num_classes] logits.

So conceptually:

“Look at the 1D sequence of offensive players and learn patterns that distinguish Deep_Pass from Short_Pass from Runs.”

train_cnn.py (inside cnn/)

This is an older in-package training script version. You’re primarily using scripts/train_cnn.py now, which does the same job but from the scripts/ folder.

You can keep it as a reference or ignore it; the main training entrypoint is your scripts/train_cnn.py.

4. Other key files
pyproject.toml

Declares your project as a Python package:

[project]
name = "Football_Play_Project"
version = "0.1.0"
dependencies = ["ultralytics", "torch", "opencv-python", ...]


Lets you install in editable mode:

pip install -e .


That’s why imports like from Football_Play_Project.cnn.dataset import PlayDataset work.

yolov8n.pt

Pretrained YOLOv8 nano model from Ultralytics.

Detects generic “person” objects.

You’re using it as:

A generic player detector (and manually cleaning).

Optionally, a base model to fine-tune with yolo_train.py for football-specific detection.