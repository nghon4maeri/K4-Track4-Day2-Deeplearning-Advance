import torch
import torch.nn as nn
import timm

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}

def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    timm_name = SUGGESTED_BACKBONES.get(name, name)
    
    model = timm.create_model(
        timm_name, 
        pretrained=pretrained if init != "scratch" else False, 
        num_classes=num_classes, 
        drop_rate=drop_rate
    )
    
    if init == "frozen":
        freeze_backbone(model)
        
    return model

def freeze_backbone(model) -> None:
    # Đóng băng toàn bộ backbone, chỉ mở requires_grad cho classifier
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = model.get_classifier()
    for param in classifier.parameters():
        param.requires_grad = True

def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    # Differentiate between weights (ndim > 1) and norm/bias (ndim <= 1)
    # Head vs Backbone
    groups = []
    
    backbone_weights = []
    backbone_bias_norm = []
    head_weights = []
    head_bias_norm = []
    
    # Identify head parameters by comparing object ids
    classifier_params = set(id(p) for p in model.get_classifier().parameters())
    
    for param in model.parameters():
        if not param.requires_grad:
            continue
            
        is_head = id(param) in classifier_params
        
        if is_head:
            if param.ndim > 1:
                head_weights.append(param)
            else:
                head_bias_norm.append(param)
        else:
            if param.ndim > 1:
                backbone_weights.append(param)
            else:
                backbone_bias_norm.append(param)
                
    if backbone_weights:
        groups.append({"params": backbone_weights, "lr": lr_backbone, "weight_decay": weight_decay})
    if backbone_bias_norm:
        groups.append({"params": backbone_bias_norm, "lr": lr_backbone, "weight_decay": 0.0})
    if head_weights:
        groups.append({"params": head_weights, "lr": lr_head, "weight_decay": weight_decay})
    if head_bias_norm:
        groups.append({"params": head_bias_norm, "lr": lr_head, "weight_decay": 0.0})
        
    return groups

def count_params(model) -> float:
    total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1e6

def count_gmacs(model, img_size: int = 224) -> float:
    # Ước lượng sử dụng thư viện fvcore (chính xác nhất theo slide)
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        flops = FlopCountAnalysis(model, dummy_input)
        macs = flops.total()
        return macs / 1e9
    except ImportError:
        try:
            from ptflops import get_model_complexity_info
            macs, _ = get_model_complexity_info(model, (3, img_size, img_size), as_strings=False, print_per_layer_stat=False, verbose=False)
            return macs / 1e9
        except ImportError:
            print("Vui lòng cài đặt `fvcore` hoặc `ptflops` để đếm GMACs chính xác.")
            return 0.0
