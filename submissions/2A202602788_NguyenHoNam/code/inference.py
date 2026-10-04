import torch
import torch.nn.functional as F
import time
import numpy as np
from typing import Optional, List

def predict_logits(model, loader, device, view=None):
    model.eval()
    all_logits = []
    all_targets = []
    all_filenames = []
    
    with torch.inference_mode():
        for inputs, targets, filenames in loader:
            inputs = inputs.to(device)
            if view:
                inputs = view(inputs)
            
            logits = model(inputs)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_filenames.extend(filenames)
            
    return all_filenames, np.concatenate(all_targets), np.concatenate(all_logits)

def view_identity(x):
    return x

def view_hflip(x):
    return torch.flip(x, dims=[3])

def views_multicrop(x, crop: int):
    _, _, h, w = x.shape
    crops = []
    # 4 corners + center
    crops.append(x[:, :, 0:crop, 0:crop])
    crops.append(x[:, :, 0:crop, w-crop:w])
    crops.append(x[:, :, h-crop:h, 0:crop])
    crops.append(x[:, :, h-crop:h, w-crop:w])
    ch, cw = h//2, w//2
    c_half = crop//2
    crops.append(x[:, :, ch-c_half:ch+c_half+(crop%2), cw-c_half:cw+c_half+(crop%2)])
    return crops

def views_multiscale(x, sizes):
    return [F.interpolate(x, size=(s, s), mode='bilinear', align_corners=False) for s in sizes]

def aggregate_views(logits_per_view, space: str = "prob"):
    # logits_per_view: List of np.ndarray shape (N, 9)
    if space == "prob":
        probs = [F.softmax(torch.tensor(lg), dim=1).numpy() for lg in logits_per_view]
        return np.mean(probs, axis=0)
    elif space == "logit":
        avg_logits = np.mean(logits_per_view, axis=0)
        return F.softmax(torch.tensor(avg_logits), dim=1).numpy()
    raise ValueError(f"Unknown space {space}")

def ensemble_probs(list_of_probs):
    return np.mean(list_of_probs, axis=0)

def fit_temperature(val_logits, val_labels) -> float:
    from scipy.optimize import minimize
    
    def nll(t):
        t = t[0]
        logits_t = val_logits / t
        probs = np.exp(logits_t) / np.sum(np.exp(logits_t), axis=1, keepdims=True)
        probs = np.clip(probs, 1e-12, 1.0)
        nll_val = -np.mean(np.log(probs[np.arange(len(val_labels)), val_labels]))
        return nll_val
        
    res = minimize(nll, [1.0], bounds=[(0.01, 10.0)])
    return float(res.x[0])

def apply_temperature(logits, T: float):
    logits_t = logits / T
    return F.softmax(torch.tensor(logits_t), dim=1).numpy()

def fuse_conv_bn(model):
    import torch.nn.utils.fuse_conv_bn_eval
    model.eval()
    try:
        model = torch.nn.utils.fuse_conv_bn_eval(model)
    except:
        pass # In case not supported
    return model
