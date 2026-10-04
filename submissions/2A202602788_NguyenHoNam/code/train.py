import os
import sys
import json
import time
import random
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Insert path to access eval.py
REPO_DIR = str(Path(__file__).resolve().parent.parent)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
from eval import save_predictions, compute_metrics

from starter.dataset import load_split, check_split, build_transforms, make_loader
from starter.model import build_model, param_groups, count_params, count_gmacs
from starter.losses import build_criterion, class_weights, mix_batch, mixed_loss

@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "resnet50"
    init: str = "finetune"
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    save_test_predictions: bool = False

def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"

def pred_path(cfg: Config, split: str) -> Path:
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_optimizer(model, cfg: Config):
    groups = param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return AdamW(groups)

def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)
    
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

class EMA:
    def __init__(self, model, decay: float):
        self.decay = decay
        self.ema_model = torch.utils.data.dataloader.default_collate([model.state_dict()])
        self.ema_model = {k: v.clone().detach() for k, v in self.ema_model[0].items()}
        
    def update(self, model):
        with torch.no_grad():
            for key, param in model.state_dict().items():
                if key in self.ema_model:
                    self.ema_model[key].copy_(self.decay * self.ema_model[key] + (1.0 - self.decay) * param.detach())

    def apply(self, model):
        model.load_state_dict(self.ema_model)

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    if cfg.init == "frozen":
        model.eval()
        model.get_classifier().train()
    else:
        model.train()
        
    total_loss = 0
    
    for inputs, targets, _ in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        if cfg.mix:
            inputs, mixed_targets = mix_batch(inputs, targets, cfg.mix_alpha, cfg.mix)
            
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            logits = model(inputs)
            if cfg.mix:
                loss = mixed_loss(criterion, logits, mixed_targets)
            else:
                loss = criterion(logits, targets)
                
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        if ema:
            ema.update(model)
            
        total_loss += loss.item()
        
    return {"train_loss": total_loss / len(loader)}

def evaluate(model, loader, criterion, device):
    model.eval()
    all_logits = []
    all_targets = []
    all_filenames = []
    total_loss = 0
    
    with torch.inference_mode():
        for inputs, targets, filenames in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            
            total_loss += loss.item()
            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_filenames.extend(filenames)
            
    return all_filenames, np.concatenate(all_targets), np.concatenate(all_logits), total_loss / len(loader)

def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(epochs, train_loss, label='Train Loss', color='tab:red', linestyle='--')
    ax1.plot(epochs, val_loss, label='Val Loss', color='tab:red')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Macro F1')
    ax2.plot(epochs, val_f1, label='Val Macro-F1', color='tab:blue')
    ax2.tick_params(axis='y')
    ax2.legend(loc='upper right')
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def run(cfg: Config) -> dict:
    set_seed(cfg.seed)
    out_dir = run_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)
    Path("curves").mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=4)
        
    train_df, val_df, test_df = load_split(cfg.labels_dir, cfg.fold)
    check_split(train_df, val_df, test_df, cfg.images_dir)
    
    train_trans = build_transforms(True, cfg.img_size, cfg.aug)
    val_trans = build_transforms(False, cfg.img_size)
    
    train_loader = make_loader(train_df, cfg.images_dir, train_trans, cfg.batch_size, True, cfg.sampler, cfg.num_workers)
    val_loader = make_loader(val_df, cfg.images_dir, val_trans, cfg.batch_size, False, None, cfg.num_workers)
    
    if cfg.save_test_predictions:
        test_loader = make_loader(test_df, cfg.images_dir, val_trans, cfg.batch_size, False, None, cfg.num_workers)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_model(cfg.backbone, True, 9, cfg.drop_rate, cfg.init).to(device)
    
    cw = class_weights(train_df['Label'].value_counts().sort_index().values, cfg.class_weight_beta or 0.0).to(device) if cfg.loss == "ce_weighted" else None
    criterion = build_criterion(cfg.loss, smoothing=cfg.label_smoothing, gamma=cfg.focal_gamma, weight=cw)
    
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
    
    history = []
    best_f1 = 0.0
    best_epoch = 0
    
    start_time = time.time()
    for epoch in range(1, cfg.epochs + 1):
        ep_start = time.time()
        train_res = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        
        eval_model = model
        if ema:
            eval_model = build_model(cfg.backbone, False, 9, cfg.drop_rate, cfg.init).to(device)
            ema.apply(eval_model)
            
        filenames, y_true, logits, val_loss = evaluate(eval_model, val_loader, criterion, device)
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
        metrics = compute_metrics(y_true, probs.argmax(axis=1), probs, 9)
        
        ep_time = time.time() - ep_start
        macro_f1 = metrics["macro_f1"]
        
        history.append({
            "epoch": epoch,
            "train_loss": train_res["train_loss"],
            "val_loss": val_loss,
            "val_macro_f1": macro_f1,
            "time": ep_time
        })
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_epoch = epoch
            torch.save(eval_model.state_dict(), out_dir / "best_model.pth")
            
    # Load best model for final evaluation
    model.load_state_dict(torch.load(out_dir / "best_model.pth"))
    
    # Save val predictions
    filenames, y_true, logits, _ = evaluate(model, val_loader, criterion, device)
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
    save_predictions(pred_path(cfg, "val"), filenames, y_true, probs)
    
    if cfg.save_test_predictions:
        filenames, y_true, logits, _ = evaluate(model, test_loader, criterion, device)
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
        save_predictions(pred_path(cfg, "test"), filenames, y_true, probs)
        
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    plot_curves(history, Path("curves") / f"{cfg.exp_id}_{cfg.backbone}.png", f"{cfg.exp_id} - {cfg.backbone}")
    
    return {
        "best_epoch": best_epoch,
        "best_macro_f1": best_f1,
        "time_per_epoch": (time.time() - start_time) / cfg.epochs,
        "params": count_params(model),
        "gmacs": count_gmacs(model, cfg.img_size)
    }

def parse_overrides(pairs: list[str]) -> dict:
    import ast
    d = {}
    for pair in pairs:
        if "=" not in pair: continue
        k, v = pair.split("=", 1)
        try:
            d[k] = ast.literal_eval(v)
        except:
            d[k] = v
    return d

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", nargs="+", help="Override config (e.g. exp_id=B01 seed=0)")
    args = parser.parse_args()
    
    overrides = parse_overrides(args.set) if args.set else {}
    cfg = Config(**overrides)
    run(cfg)

if __name__ == "__main__":
    main()
