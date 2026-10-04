import torch
from starter.model import build_model
from starter.benchmark import latency_report, tta_latency
import pandas as pd

def run_latency_tests():
    print("Running latency tests on ResNet50...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Baseline FP32
    model = build_model("resnet50", pretrained=False, num_classes=9).to(device)
    res_base = latency_report(model, batch_size=1, img_size=224, dtype="fp32", device=device)
    print("Baseline FP32:", res_base)
    
    # 2. FP16 (AMP)
    res_amp = latency_report(model, batch_size=1, img_size=224, dtype="fp16", device=device)
    print("FP16:", res_amp)
    
    # 3. Fuse Conv-BN
    from starter.inference import fuse_conv_bn
    model_fused = fuse_conv_bn(model)
    res_fused = latency_report(model_fused, batch_size=1, img_size=224, dtype="fp32", device=device)
    print("Fused Conv-BN:", res_fused)
    
    # 4. TTA x5
    res_tta = tta_latency(model, k_views=5, batch_size=1, img_size=224, dtype="fp32", device=device)
    print("TTA 5-views:", res_tta)

    # Note: F1 vs Latency requires actually running evaluation with TTA, which takes a while.
    # We will log these numbers for the report.

if __name__ == "__main__":
    run_latency_tests()
