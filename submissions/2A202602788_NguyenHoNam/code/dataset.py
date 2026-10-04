import os
from pathlib import Path
from typing import Tuple, Dict, Optional
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

NUM_CLASSES = 9
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def load_split(labels_dir: str | Path, fold: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels_dir = Path(labels_dir)
    train_df = pd.read_csv(labels_dir / f"train_subset{fold}.csv")
    val_df = pd.read_csv(labels_dir / f"val_subset{fold}.csv")
    test_df = pd.read_csv(labels_dir / f"test_subset{fold}.csv")
    return train_df, val_df, test_df

def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    images_dir = Path(images_dir)
    
    train_counts = train_df['Label'].value_counts().to_dict()
    val_counts = val_df['Label'].value_counts().to_dict()
    test_counts = test_df['Label'].value_counts().to_dict()
    
    train_set = set(train_df['Filename'])
    val_set = set(val_df['Filename'])
    test_set = set(test_df['Filename'])
    
    overlap_train_val = train_set.intersection(val_set)
    overlap_train_test = train_set.intersection(test_set)
    overlap_val_test = val_set.intersection(test_set)
    
    assert len(overlap_train_val) == 0, "Lỗi: Train và Val bị trùng ảnh!"
    assert len(overlap_train_test) == 0, "Lỗi: Train và Test bị trùng ảnh!"
    assert len(overlap_val_test) == 0, "Lỗi: Val và Test bị trùng ảnh!"
    
    total_imgs = len(train_set.union(val_set).union(test_set))
    assert total_imgs == 17509, f"Lỗi: Tổng số ảnh là {total_imgs}, mong đợi 17509"
        
    stats = {
        "n": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "per_class": {"train": train_counts, "val": val_counts, "test": test_counts},
        "overlap": {"train_val": len(overlap_train_val), "train_test": len(overlap_train_test), "val_test": len(overlap_val_test)}
    }
    return stats

def get_resize(h, w):
    try:
        return A.Resize(height=h, width=w)
    except:
        return A.Resize(size=(h, w))

def get_center_crop(h, w):
    try:
        return A.CenterCrop(height=h, width=w)
    except:
        return A.CenterCrop(size=(h, w))

def get_random_resized_crop(h, w):
    try:
        return A.RandomResizedCrop(size=(h, w), scale=(0.08, 1.0))
    except:
        return A.RandomResizedCrop(height=h, width=w, scale=(0.08, 1.0))

def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    if train:
        if aug == "basic":
            return A.Compose([
                get_random_resized_crop(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ])
        elif aug == "color":
            return A.Compose([
                get_random_resized_crop(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ])
        else: # randaug/trivial - strong aug
            return A.Compose([
                get_random_resized_crop(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ])
    else:
        return A.Compose([
            get_resize(256, 256),
            get_center_crop(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

class DeepWeedsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        self.filenames = self.df['Filename'].values
        self.labels = self.df['Label'].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        filename = self.filenames[i]
        label = int(self.labels[i])
        img_path = self.images_dir / filename
        
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        
        if self.transform:
            augmented = self.transform(image=img_np)
            tensor = augmented['image']
        else:
            tensor = torch.from_numpy(img_np.transpose((2, 0, 1))).float() / 255.0
            
        return tensor, label, filename

def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    dataset = DeepWeedsDataset(df, images_dir, transform)
    
    if train and sampler == "balanced":
        class_counts = df['Label'].value_counts().sort_index().values
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[df['Label'].values]
        sampler_obj = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler_obj,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    elif train:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True, drop_last=False)
