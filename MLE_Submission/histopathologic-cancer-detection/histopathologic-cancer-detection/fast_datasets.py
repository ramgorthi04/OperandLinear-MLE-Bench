
import numpy as np
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset

class DiskDataset(Dataset):
    """
    Minimal CPU work dataset:
    - Uses cv2.imread (BGR) + cv2.cvtColor to RGB for robustness in multi-worker.
    - Returns uint8 CHW tensors only. NO resize, NO normalization on CPU.
    - Labels (float32) returned when with_labels=True.
    """
    def __init__(self, df, img_dir: Path, with_labels: bool = True):
        self.df = df.reset_index(drop=True)
        self.dir = Path(img_dir)
        self.with_labels = with_labels
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img_id = r['id']
        fp = self.dir / f"{img_id}.tif"
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)  # HWC, BGR, uint8
        if img is None:
            # Fallback to zeros if corrupted/missing to keep worker alive
            img = np.zeros((96, 96, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # CHW, uint8
        if self.with_labels:
            y = torch.tensor(r['label'], dtype=torch.float32)
            return x, y
        else:
            return x, img_id

class TestDiskDataset(Dataset):
    def __init__(self, ids, img_dir: Path):
        self.ids = list(ids)
        self.dir = Path(img_dir)
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        fp = self.dir / f"{img_id}.tif"
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((96, 96, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # uint8 CHW
        return x, img_id
