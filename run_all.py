import os
import subprocess
import time

def run_cmd(cmd):
    print(f"\n[{time.strftime('%X')}] Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    # Cấu hình
    epochs = 10
    
    # Bước 1: Train Backbone
    run_cmd(f"python starter/train.py --set exp_id=B01 backbone=resnet50 epochs={epochs}")
    run_cmd(f"python starter/train.py --set exp_id=B02 backbone=convnext_tiny epochs={epochs}")
    
    # Bước 2: Tuning
    run_cmd(f"python starter/train.py --set exp_id=T01 backbone=resnet50 loss=focal focal_gamma=2.0 epochs={epochs}")
    run_cmd(f"python starter/train.py --set exp_id=T02 backbone=resnet50 mix=cutmix mix_alpha=1.0 epochs={epochs}")
    
    # Bước 4: Final 3 Seeds (Dùng CutMix vì thường cho F1 tốt)
    for seed in [0, 1, 2]:
        run_cmd(f"python starter/train.py --set exp_id=F01 backbone=resnet50 mix=cutmix seed={seed} epochs={epochs} save_test_predictions=True")
        run_cmd(f"python starter/train.py --set exp_id=T00 backbone=resnet50 seed={seed} epochs={epochs} save_test_predictions=True")
        
    # Đánh giá tự động
    print("\nCalculating scores with eval.py...")
    run_cmd("python eval.py score --pred \"predictions/F01_seed*_test.csv\" --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01 --out eval_out")
    run_cmd("python eval.py grade --final \"predictions/F01_seed*_test.csv\" --baseline \"predictions/T00_seed*_test.csv\" --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --out eval_out")
    
    print("\n>>> ALL PIPELINES COMPLETED! <<<")

if __name__ == "__main__":
    main()
