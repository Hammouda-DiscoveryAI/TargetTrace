"""Quick profiling script — run with: python profile_train.py"""
import time, torch, numpy as np
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

from database import get_training_data
from sklearn.preprocessing import LabelEncoder
from trainer import TargetTraceDataset, device, USE_AMP, _collate
from torch.utils.data import DataLoader
from model import TargetTrace
import torch.nn as nn
from torch.amp import autocast, GradScaler

SAMPLE = 50_000
FULL   = 2_614_433

df = get_training_data()
df = df[df["target_name"].notna()].reset_index(drop=True)
df_s = df.sample(SAMPLE, random_state=42).reset_index(drop=True)
le = LabelEncoder(); le.fit(df_s["target_name"].unique())
p_mean = float(df_s["pic50"].dropna().mean())
p_std  = max(float(df_s["pic50"].dropna().std()), 0.1)

# ── Phase 1: Dataset init ──────────────────────────────────────────────────
print("=== Phase 1: Dataset.__init__ ===")
t0 = time.time()
ds = TargetTraceDataset(df_s, le, p_mean, p_std, augment=True)
dt = time.time() - t0
print(f"  {SAMPLE//1000}k rows:  {dt:.2f}s")
print(f"  Extrapolated to {FULL/1e6:.1f}M: {dt*FULL/SAMPLE:.0f}s  ({dt*FULL/SAMPLE/60:.1f} min)")
print(f"  Unique SMILES in subset: {len(ds._fp):,}")

# ── Phase 2: Single __getitem__ ────────────────────────────────────────────
print("\n=== Phase 2: __getitem__ latency (1000 samples) ===")
t0 = time.time()
for i in range(1000):
    _ = ds[i]
item_us = (time.time() - t0) / 1000 * 1e6
print(f"  Avg per item: {item_us:.1f} µs")
print(f"  Full epoch collation overhead: {item_us*FULL/1e6/60:.1f} min")

# ── Phase 3: GPU forward+backward ─────────────────────────────────────────
print("\n=== Phase 3: GPU forward+backward (batch=128) ===")
loader = DataLoader(ds, batch_size=128, shuffle=True, collate_fn=_collate, num_workers=0)
model  = TargetTrace(len(le.classes_)).to(device)
opt    = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler(enabled=USE_AMP)
cls_fn = nn.CrossEntropyLoss()
model.train()
times = []
for i, b in enumerate(loader):
    if i >= 22: break
    t0 = time.time()
    fp,bert,gx,gadj,amask,ep,er,pm,lbl,_ = [x.to(device, non_blocking=True) for x in b]
    opt.zero_grad(set_to_none=True)
    with autocast(device_type="cuda", enabled=USE_AMP):
        tl, pa = model(fp, bert, gx, gadj, amask, ep, er, pm)
        loss = cls_fn(tl, lbl)
    scaler.scale(loss).backward()
    scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    times.append(time.time() - t0)

avg = np.mean(times[2:]) * 1000
full_batches = FULL // 128
print(f"  Avg batch time (post-warmup): {avg:.1f}ms")
print(f"  Full epoch ({full_batches:,} batches): {avg/1000*full_batches/60:.1f} min")
print(f"  10 epochs:                            {avg/1000*full_batches*10/60:.0f} min")
print(f"\n  GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"  GPU reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")

# ── Phase 4: Bottleneck breakdown ─────────────────────────────────────────
print("\n=== Summary ===")
print(f"  Dataset init (once per run): {dt*FULL/SAMPLE:.0f}s ({dt*FULL/SAMPLE/60:.1f} min)")
print(f"  Per-epoch GPU time:          {avg/1000*full_batches/60:.1f} min")
print(f"  Bottleneck: {'Dataset init' if dt*FULL/SAMPLE > avg/1000*full_batches else 'GPU training loop'}")
