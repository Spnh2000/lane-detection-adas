import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model  import LaneNet
from utils  import TuSimpleDataset

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_ROOT  = "data/tusimple/train_set"
JSON_FILES = [
    "data/tusimple/train_set/label_data_0313.json",
    "data/tusimple/train_set/label_data_0531.json",
    "data/tusimple/train_set/label_data_0601.json",
]
IMG_SIZE   = (256, 512)
BATCH_SIZE = 8
EPOCHS     = 20
LR         = 1e-3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH  = "outputs/lanenet_best.pth"

def dice_loss(pred, target, smooth=1.0):
    pred   = torch.sigmoid(pred)
    inter  = (pred * target).sum(dim=(2, 3))
    union  = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - ((2 * inter + smooth) / (union + smooth)).mean()

def train():
    os.makedirs("outputs", exist_ok=True)
    print(f"Using device: {DEVICE}")

    # Dataset
    dataset = TuSimpleDataset(JSON_FILES, DATA_ROOT, IMG_SIZE)
    val_len = int(len(dataset) * 0.1)
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_len, val_len])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Model
    model     = LaneNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    bce       = nn.BCEWithLogitsLoss()

    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0
        for imgs, masks in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = bce