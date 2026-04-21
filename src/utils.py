import json
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

class TuSimpleDataset(Dataset):
    """
    Loads TuSimple lane detection dataset.
    Returns (image_tensor, lane_mask_tensor) pairs.
    """
    def __init__(self, json_paths, img_root, img_size=(256, 512)):
        self.img_size = img_size  # (H, W)
        self.img_root = img_root
        self.samples  = []

        for json_path in json_paths:
            with open(json_path) as f:
                for line in f:
                    self.samples.append(json.loads(line))

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std= [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample   = self.samples[idx]
        img_path = os.path.join(self.img_root, sample["raw_file"])

        # ── Image ──────────────────────────────────────────────────────────────
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))

        # ── Lane mask from keypoints ────────────────────────────────────────────
        mask = self._make_mask(sample)

        img_tensor  = self.transform(img)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float