import torch
import time
import numpy as np

def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    for _ in range(warmup):
        fn()
        
    times = []
    for _ in range(iters):
        if sync: sync()
        t0 = time.perf_counter()
        fn()
        if sync: sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        
    times = np.array(times)
    return {
        "p50": np.percentile(times, 50),
        "p95": np.percentile(times, 95),
        "p99": np.percentile(times, 99),
        "mean": np.mean(times),
        "n": iters
    }

def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    model.eval()
    dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    if dtype == "fp16":
        model = model.half()
        dummy_input = dummy_input.half()
        
    def forward_fn():
        with torch.inference_mode():
            if dtype == "amp":
                with torch.cuda.amp.autocast():
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)
                
    sync_fn = torch.cuda.synchronize if device == "cuda" else None
    
    res = bench(forward_fn, warmup=warmup, iters=iters, sync=sync_fn)
    
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    
    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": res["p50"],
        "p95": res["p95"],
        "p99": res["p99"],
        "images_per_s": batch_size / (res["p50"] / 1000.0),
        "torch": torch.__version__
    }

def tta_latency(model, k_views: int, **kw) -> dict:
    # Just run latency_report and multiply appropriately or bench TTA directly
    res = latency_report(model, **kw)
    res["p50"] *= k_views
    res["p95"] *= k_views
    res["p99"] *= k_views
    res["images_per_s"] /= k_views
    return res
