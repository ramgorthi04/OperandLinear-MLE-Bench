
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset

def pil_read_rgb(path: Path):
    with Image.open(path) as im:
        return np.array(im.convert('RGB'))

def preload_images_to_ram(ids, img_dir: Path, desc='preload', log_every=5000):
    cache = {}
    t0 = time.time()
    for i, img_id in enumerate(ids):
        cache[img_id] = pil_read_rgb(img_dir / f"{img_id}.tif")
        if log_every and (i+1) % log_every == 0:
            print(f"{desc}: {i+1}/{len(ids)} loaded ({time.time()-t0:.1f}s)")
    print(f"{desc}: loaded {len(ids)} images to RAM in {time.time()-t0:.1f}s")
    return cache

class HistoDataset(Dataset):
    def __init__(self, df, image_cache, transforms=None):
        self.df = df.reset_index(drop=True)
        self.image_cache = image_cache
        self.transforms = transforms
        self.has_label = 'label' in df.columns
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = self.image_cache[r['id']]
        if self.transforms:
            img = self.transforms(image=img)['image']
        if self.has_label:
            label = torch.tensor(r['label'], dtype=torch.float32)
            return img, label
        else:
            return img, r['id']
