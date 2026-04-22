cat > src/train.py << 'EOF'
import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from model import LaneNet
from utils import TuSimpleDataset

parser = argparse.ArgumentParser()
parser.add_argument("--data_root",  default="data/tusimple/train_set")
parser.add_argument("--json_files", nargs="+", default=[
    "data/tusimple/train_set/label_data_0313.json",
    "data/tusimple/train_set/label_data_0531.json",
    "data/tusimple/train_set/label_data_0601.json",
])
parser.add_argument("--epochs",     type=int,   default=20)
parser.add_argument("--batch_size", type=int,   default=8)
parser.add_argument("--lr",         type=float, default=1e-3)
args = parser.parse_args()

DATA_ROOT  = args.data_root
JSON_FILES = args.json_files
IMG_SIZE   = (256, 512)
BATCH_SIZE = args.batch_size
EPOCHS     = args.epochs
LR         = args.lr
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH  = "outputs/lanenet_best.pth"

def dice_loss(pred, target, smooth=1.0):
    pred  = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - ((2 * inter + smooth) / (union + smooth)).mean()

def train():
    os.makedirs("outputs", exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Loading dataset from {DATA_ROOT}...")

    dataset = TuSimpleDataset(JSON_FILES, DATA_ROOT, IMG_SIZE)
    val_len = int(len(dataset) * 0.1)
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_len, val_len])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model     = LaneNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    bce       = nn.BCEWithLogitsLoss()
    best_val  = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for imgs, masks in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = bce(preds, masks) + dice_loss(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs)
                val_loss += (bce(preds, masks) + dice_loss(preds, masks)).item()

        train_loss /= len(train_dl)
        val_loss   /= len(val_dl)
        scheduler.step(val_loss)
        print(f"Epoch {epoch}/{EPOCHS} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✓ Saved best model → {SAVE_PATH}")

    print(f"Done! Best val loss: {best_val:.4f}")

if __name__ == "__main__":
    train()
EOF