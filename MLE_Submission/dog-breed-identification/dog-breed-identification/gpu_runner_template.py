#!/usr/bin/env python3
import os, json, math, time, random, argparse
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image, ImageOps
import timm
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2
from timm.data import Mixup

SEED = 20250810
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
assert device == 'cuda', 'GPU not available. Please enable a GPU runtime.'

BASE = Path('.')
labels_df = pd.read_csv('labels.csv')
folds_df = pd.read_csv('fold_assignments.csv')  # must match labels_df order
test_meta = pd.read_csv('test_image_meta.csv')
sample_df = pd.read_csv('sample_submission.csv')
classes = [c for c in sample_df.columns if c != 'id']
breed_to_idx = {b:i for i,b in enumerate(classes)}
train_ids = labels_df['id'].tolist()
test_ids  = test_meta['id'].tolist()
y_all = labels_df['breed'].map(breed_to_idx).values.astype(np.int64)
folds = folds_df.set_index('id').loc[labels_df['id'], 'fold'].values.astype(int)
n_classes = len(classes)
n_folds = int(folds.max() + 1)

class DogDataset(Dataset):
    def __init__(self, ids, split: str, labels: Optional[np.ndarray] = None, img_size: int = 384, aug: bool = False):
        self.ids = ids; self.split = split; self.labels = labels
        self.root = Path('train' if split=='train' else 'test')
        if aug:
            self.tf = T.Compose([
                T.Resize(int(img_size*1.1), interpolation=T.InterpolationMode.BICUBIC),
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.5), T.RandAugment(num_ops=2, magnitude=9),
                T.ToTensor(), T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)), T.RandomErasing(p=0.25, value='random')
            ])
        else:
            self.tf = T.Compose([
                T.Resize(int(img_size*1.1), interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(img_size), T.ToTensor(), T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
            ])
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        id_ = self.ids[i]
        img = Image.open(self.root / f"{id_}.jpg"); img = ImageOps.exif_transpose(img)
        if img.mode != 'RGB': img = img.convert('RGB')
        x = self.tf(img); img.close()
        if self.labels is None: return x, id_
        return x, self.labels[i]

def mixup_fn(num_classes: int):
    return Mixup(mixup_alpha=0.2, cutmix_alpha=0.1, prob=1.0, switch_prob=0.5, mode='batch', label_smoothing=0.05, num_classes=num_classes)

def create_model(model_name: str, num_classes: int, drop_path: float = 0.2):
    return timm.create_model(model_name, pretrained=True, num_classes=num_classes, drop_path_rate=drop_path)

def extract_feats(m: nn.Module, xb: torch.Tensor) -> torch.Tensor:
    if hasattr(m, 'forward_features'):
        feats = m.forward_features(xb)
    else:
        feats = m(xb)
    if feats.ndim == 4:
        feats = feats.mean(dim=(2,3))
    return feats

def scheduler_warm_cos(opt, epochs, iters_per_epoch, warmup_epochs=2):
    total = max(1, epochs*iters_per_epoch); warm = int(max(0, warmup_epochs)*iters_per_epoch)
    def lr_lambda(step):
        if warm > 0 and step < warm:
            return float(step+1)/float(max(1,warm))
        prog = 0.0 if total==warm else float(step-warm)/float(max(1,total-warm))
        return 0.5*(1.0+math.cos(math.pi*min(1.0,max(0.0,prog))))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

def make_tta(scales: List[int]):
    fns=[]
    for sz in scales:
        def rf(x, s=sz):
            if x.shape[-1]==s and x.shape[-2]==s: return x
            return F.interpolate(x, size=(s,s), mode='bilinear', align_corners=False)
        fns.append(lambda x, r=rf: r(x))
        fns.append(lambda x, r=rf: torch.flip(r(x), dims=[3]))
    return fns

def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    dev='cuda'
    T = torch.tensor(1.0, requires_grad=True, device=dev)
    x = torch.from_numpy(logits).to(dev)
    y = torch.from_numpy(labels).long().to(dev)
    nll = nn.CrossEntropyLoss()
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=100, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); loss = nll(x/torch.clamp(T, min=1e-3), y); loss.backward(); return loss
    opt.step(closure)
    return float(T.detach().float().clamp_min(1e-3).cpu().item())

def train_fold(fold:int, model_name='convnext_base', img_size=384, epochs=8, bs=32, lr=1e-3, wd=2e-2, num_workers=8, warmup_epochs=2, tta_scales: Optional[List[int]] = None):
    tr = np.where(folds!=fold)[0]; va = np.where(folds==fold)[0]
    ids_tr = [train_ids[i] for i in tr]; ids_va = [train_ids[i] for i in va]
    y_tr = y_all[tr]; y_va = y_all[va]
    ds_tr = DogDataset(ids_tr, 'train', y_tr, img_size, aug=True)
    ds_va = DogDataset(ids_va, 'train', y_va, img_size, aug=False)
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    dl_va = DataLoader(ds_va, batch_size=bs*2, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    model = create_model(model_name, n_classes, 0.2).to(device)
    ema = ModelEmaV2(model, decay=0.9998, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = scheduler_warm_cos(opt, epochs, max(1,len(dl_tr)), warmup_epochs=warmup_epochs)
    mx = mixup_fn(n_classes); crit = SoftTargetCrossEntropy(); scaler = torch.cuda.amp.GradScaler()
    best_loss = 1e9; best_state=None
    for ep in range(epochs):
        model.train(); t0=time.time()
        for xb, yb in dl_tr:
            xb=xb.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True)
            xb, ybm = mx(xb, yb); opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits = model(xb); loss = crit(logits, ybm)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); ema.update(model); sched.step()
        # val
        ema_m = ema.module; ema_m.eval(); vl=0.0; n=0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb=xb.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True)
                with torch.cuda.amp.autocast():
                    l = ema_m(xb); loss = nn.CrossEntropyLoss()(l, yb)
                vl += loss.item()*xb.size(0); n += xb.size(0)
        vl /= max(1,n)
        if vl < best_loss - 1e-6:
            best_loss = vl; best_state = {k: v.detach().cpu() for k,v in ema_m.state_dict().items()}
        print(f'Fold {fold} Epoch {ep+1}/{epochs} val_loss={vl:.5f}')
    if best_state is not None:
        ema_m.load_state_dict(best_state, strict=False)
    # OOF logits/embeds for validation indices
    oof_logits_f = np.zeros((len(va), n_classes), dtype=np.float32)
    oof_embeds_f = None
    ema_m.eval();
    with torch.no_grad():
        ptr=0
        for xb, yb in dl_va:
            xb=xb.to(device,non_blocking=True)
            l = ema_m(xb); f = extract_feats(ema_m, xb)
            lnp = l.float().cpu().numpy(); fnp = f.float().cpu().numpy()
            if oof_embeds_f is None: oof_embeds_f = np.zeros((len(va), fnp.shape[1]), dtype=np.float32)
            oof_logits_f[ptr:ptr+lnp.shape[0]] = lnp; oof_embeds_f[ptr:ptr+fnp.shape[0]] = fnp; ptr += lnp.shape[0]
    # Test-time logits/embeds with parameterized multi-scale TTA (with hflip)
    if tta_scales is None or len(tta_scales) == 0:
        # Dynamic default based on training img_size
        tta_scales = [int(img_size), int(round(img_size * 1.15))]
    tta_fns = make_tta(tta_scales)
    ds_te = DogDataset(test_ids, 'test', labels=None, img_size=img_size, aug=False)
    dl_te = DataLoader(ds_te, batch_size=bs*2, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    test_logits_f = np.zeros((len(test_ids), n_classes), dtype=np.float32)
    test_embeds_f = None
    with torch.no_grad():
        ofs=0
        for xb,_ in dl_te:
            xb=xb.to(device,non_blocking=True)
            acc_l=None; acc_f=None
            for tta in tta_fns:
                xa = tta(xb)
                la = ema_m(xa); fa = extract_feats(ema_m, xa)
                acc_l = la if acc_l is None else (acc_l+la)
                acc_f = fa if acc_f is None else (acc_f+fa)
            lnp = (acc_l / len(tta_fns)).float().cpu().numpy(); fnp = (acc_f / len(tta_fns)).float().cpu().numpy()
            nb = lnp.shape[0]
            if test_embeds_f is None: test_embeds_f = np.zeros((len(test_ids), fnp.shape[1]), dtype=np.float32)
            test_logits_f[ofs:ofs+nb] = lnp; test_embeds_f[ofs:ofs+nb] = fnp; ofs += nb
    return va, oof_logits_f, test_logits_f, float(best_loss), oof_embeds_f, test_embeds_f

def softmax_np(x):
    m = x.max(axis=1, keepdims=True); z = np.exp(x-m)
    return z / (z.sum(axis=1, keepdims=True)+1e-12)

def row_normalize(p):
    p = np.clip(p, 1e-8, 1.0); p /= p.sum(axis=1, keepdims=True); return p

def parse_scales_arg(val: Optional[str]) -> Optional[List[int]]:
    if val is None:
        return None
    s = val.strip()
    if not s:
        return None
    try:
        parts = [int(x) for x in s.replace(' ', '').split(',') if x]
        return parts if parts else None
    except Exception:
        print('WARNING: Failed to parse --tta_scales; using dynamic defaults.')
        return None

def parse_args():
    ap = argparse.ArgumentParser(description='GPU runner for Dog Breed Identification (5-fold)')
    ap.add_argument('--model_name', type=str, default='convnext_base')
    ap.add_argument('--img_size', type=int, default=384)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=2e-2)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--warmup_epochs', type=int, default=2)
    ap.add_argument('--tta_scales', type=str, default=None, help='Comma-separated scales for TTA, e.g., "384,448,512". If omitted, defaults to [img_size, round(img_size*1.15)].')
    return ap.parse_args()

def main():
    args = parse_args()
    tta_scales = parse_scales_arg(args.tta_scales)
    cfg = {'model_name': args.model_name, 'img_size': args.img_size, 'epochs': args.epochs, 'bs': args.bs, 'lr': args.lr, 'wd': args.wd, 'num_workers': args.num_workers, 'warmup_epochs': args.warmup_epochs, 'tta_scales': (tta_scales if tta_scales is not None else [int(args.img_size), int(round(args.img_size*1.15))])}
    print('GPU runner config:', cfg)
    oof_logits = np.zeros((len(train_ids), n_classes), dtype=np.float32)
    test_logits_stack = np.zeros((n_folds, len(test_ids), n_classes), dtype=np.float32)
    oof_embeds = None; test_embeds_folds = []
    fold_losses = []
    for f in range(n_folds):
        va_idx, oof_f, te_f, vloss, oof_e_f, te_e_f = train_fold(f, model_name=args.model_name, img_size=args.img_size, epochs=args.epochs, bs=args.bs, lr=args.lr, wd=args.wd, num_workers=args.num_workers, warmup_epochs=args.warmup_epochs, tta_scales=cfg['tta_scales'])
        if oof_embeds is None: oof_embeds = np.zeros((len(train_ids), oof_e_f.shape[1]), dtype=np.float32)
        oof_logits[va_idx] = oof_f; oof_embeds[va_idx] = oof_e_f
        test_logits_stack[f] = te_f; test_embeds_folds.append(te_e_f)
        fold_losses.append(vloss)
        # Intermediate saves for robustness
        np.save('oof_logits_fullimg.npy', oof_logits)
        np.save('test_logits_fullimg.npy', test_logits_stack.mean(axis=0))
        np.save('oof_embeds_fullimg.npy', oof_embeds)
        np.save('test_embeds_fullimg.npy', np.mean(np.stack(test_embeds_folds, axis=0), axis=0))
        print(f'Fold {f} done. Best val loss={vloss:.5f}')
    with open('run_manifest_convnext_base_sz384_seed20250810.json','w') as f:
        json.dump({'seed':SEED,'device':device,'classes':n_classes,'folds':n_folds,'config':cfg,'fold_val_losses':fold_losses}, f)
    # Temperature on OOF, save manifest
    T_full = fit_temperature(oof_logits, y_all)
    with open('temperatures_fullimg.json','w') as f:
        json.dump({'global_T': float(T_full), **cfg}, f)
    print(f'Calibrated temperature (OOF): T={T_full:.4f}')
    # Optional calibrated fullimg submission (diagnostic)
    te_logits = test_logits_stack.mean(axis=0)
    P = row_normalize(softmax_np(te_logits / max(1e-3, T_full)))
    sub = pd.DataFrame(P, columns=classes); sub.insert(0,'id', test_ids)
    sub.to_csv('submission_fullimg.csv', index=False)
    # Final sanity prints
    print('Artifacts written:')
    print(' - oof_logits_fullimg.npy', oof_logits.shape, oof_logits.dtype)
    print(' - test_logits_fullimg.npy', te_logits.shape, te_logits.dtype)
    print(' - oof_embeds_fullimg.npy', oof_embeds.shape, oof_embeds.dtype)
    print(' - test_embeds_fullimg.npy', np.mean(np.stack(test_embeds_folds,0),0).shape)
    print(' - temperatures_fullimg.json',)

if __name__ == '__main__':
    main()
