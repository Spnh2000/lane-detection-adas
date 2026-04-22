cat > src/utils.py << 'EOF'
import json
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

class TuSimpleDataset(Dataset):
    def __init__(self, json_paths, img_root, img_size=(256, 512)):
        self.img_size = img_size
        self.img_root = img_root
        self.samples  = []

        for json_path in json_paths:
            with open(json_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
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

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))

        mask = self._make_mask(sample)

        img_tensor  = self.transform(img)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return img_tensor, mask_tensor

    def _make_mask(self, sample):
        H, W     = self.img_size
        orig_h   = sample["h_samples"]
        lanes    = sample["lanes"]
        mask     = np.zeros((H, W), dtype=np.float32)

        x_scale  = W / 1280.0
        y_scale  = H / 720.0

        for lane in lanes:
            points = [
                (int(x * x_scale), int(y * y_scale))
                for x, y in zip(lane, orig_h)
                if x != -2
            ]
            for i in range(len(points) - 1):
                cv2.line(mask, points[i], points[i+1],
                         color=1.0, thickness=5)
        return mask
EOF