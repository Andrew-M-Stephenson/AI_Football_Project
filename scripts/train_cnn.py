# scripts/train_cnn.py

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from Football_Play_Project.cnn.dataset import PlayDataset
from Football_Play_Project.cnn.model import PlayCNN


def main():
    ap = argparse.ArgumentParser(description="Train CNN to predict plays from pre-snap structure.")
    ap.add_argument("--meta", default="data/meta/game1_people_formations.json",
                    help="Meta JSON with roles/formations.")
    ap.add_argument("--labels", default="data/meta/play_labels.csv",
                    help="CSV with key_frame_idx,play_label.")
    ap.add_argument("--max_players", type=int, default=11,
                    help="Max offensive players per play.")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--out_dir", default="runs/cnn",
                    help="Where to save model + label mapping.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Dataset
    dataset = PlayDataset(
        meta_path=args.meta,
        labels_csv=args.labels,
        max_players=args.max_players,
    )

    # Train/val split
    n_total = len(dataset)
    n_val = int(max(1, round(args.val_frac * n_total))) if n_total > 4 else 1
    n_train = n_total - n_val
    if n_train <= 0:
        raise RuntimeError("Not enough plays for train/val split.")

    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    print(f"Train plays: {len(train_ds)} | Val plays: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 2) Model
    model = PlayCNN(
        feature_dim=dataset.feature_dim,
        num_classes=dataset.num_classes,
        max_players=dataset.max_players,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 3) Training loop
    best_val_acc = 0.0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += X.size(0)

        train_loss = total_loss / max(total_samples, 1)
        train_acc = total_correct / max(total_samples, 1)

        # ----- validation -----
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                logits = model(X)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += X.size(0)

        val_acc = val_correct / max(val_total, 1)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.3f} | "
            f"val_acc={val_acc:.3f}"
        )

        # Save best model
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            ckpt_path = out_dir / "model.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label2idx": dataset.label2idx,
                    "idx2label": dataset.idx2label,
                    "feature_dim": dataset.feature_dim,
                    "max_players": dataset.max_players,
                },
                ckpt_path,
            )
            print(f"  -> Saved new best model to {ckpt_path} (val_acc={val_acc:.3f})")

    print(f"Training finished. Best val_acc={best_val_acc:.3f}")


if __name__ == "__main__":
    main()
