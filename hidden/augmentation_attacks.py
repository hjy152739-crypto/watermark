# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
模拟攻击模块，包含各种图像变换攻击，适配stable_signature项目
"""

import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import kornia


class BaseAttack(nn.Module):
    """攻击基类"""
    def __init__(self):
        super().__init__()
    
    def forward(self, imgs, masks):
        """
        Args:
            imgs: torch.Tensor [B, C, H, W] - 输入图像
            masks: torch.Tensor [B, 1, H, W] or [B, K, H, W] - 掩码
        Returns:
            tuple: (attacked_imgs, attacked_masks)
        """
        raise NotImplementedError


class IdentityAttack(BaseAttack):
    """恒等变换（无攻击）"""
    def forward(self, imgs, masks):
        return imgs, masks


class RotationAttack(BaseAttack):
    """旋转攻击"""
    def __init__(self, min_angle=-30, max_angle=30):
        super().__init__()
        self.min_angle = min_angle
        self.max_angle = max_angle
    
    def forward(self, imgs, masks):
        angle = random.uniform(self.min_angle, self.max_angle)
        
        # 旋转图像
        imgs_rotated = kornia.geometry.rotate(imgs, torch.tensor([angle] * imgs.shape[0], device=imgs.device))
        
        # 旋转掩码
        if len(masks.shape) == 4:  # [B, C, H, W]
            masks_rotated = kornia.geometry.rotate(masks.float(), torch.tensor([angle] * masks.shape[0], device=masks.device))
        else:  # [B, K, H, W]
            B, K, H, W = masks.shape
            masks_flat = masks.view(B * K, 1, H, W)
            masks_rotated = kornia.geometry.rotate(masks_flat.float(), torch.tensor([angle] * (B * K), device=masks.device))
            masks_rotated = masks_rotated.view(B, K, H, W)
        
        return imgs_rotated, masks_rotated


class ResizeAttack(BaseAttack):
    """缩放攻击"""
    def __init__(self, min_scale=0.5, max_scale=1.5):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
    
    def forward(self, imgs, masks):
        scale = random.uniform(self.min_scale, self.max_scale)
        B, C, H, W = imgs.shape
        new_h, new_w = int(H * scale), int(W * scale)
        
        # 缩放图像
        imgs_resized = F.interpolate(imgs, size=(new_h, new_w), mode='bilinear', align_corners=False)
        # 缩放回原始尺寸
        imgs_resized = F.interpolate(imgs_resized, size=(H, W), mode='bilinear', align_corners=False)
        
        # 缩放掩码
        if len(masks.shape) == 4 and masks.shape[1] == 1:  # [B, 1, H, W]
            masks_resized = F.interpolate(masks, size=(new_h, new_w), mode='nearest')
            masks_resized = F.interpolate(masks_resized, size=(H, W), mode='nearest')
        else:  # [B, K, H, W]
            B, K, H_m, W_m = masks.shape
            masks_flat = masks.view(B * K, 1, H_m, W_m)
            masks_resized = F.interpolate(masks_flat, size=(int(H_m * scale), int(W_m * scale)), mode='nearest')
            masks_resized = F.interpolate(masks_resized, size=(H_m, W_m), mode='nearest')
            masks_resized = masks_resized.view(B, K, H_m, W_m)
        
        return imgs_resized, masks_resized


class CropAttack(BaseAttack):
    """裁剪攻击"""
    def __init__(self, min_crop=0.7, max_crop=0.9):
        super().__init__()
        self.min_crop = min_crop
        self.max_crop = max_crop
    
    def forward(self, imgs, masks):
        crop_ratio = random.uniform(self.min_crop, self.max_crop)
        B, C, H, W = imgs.shape
        
        new_h = int(H * crop_ratio)
        new_w = int(W * crop_ratio)
        
        # 随机选择裁剪位置
        top = random.randint(0, H - new_h)
        left = random.randint(0, W - new_w)
        
        # 裁剪图像
        imgs_cropped = imgs[:, :, top:top+new_h, left:left+new_w]
        # 恢复原始尺寸
        imgs_cropped = F.interpolate(imgs_cropped, size=(H, W), mode='bilinear', align_corners=False)
        
        # 裁剪掩码
        if len(masks.shape) == 4 and masks.shape[1] == 1:  # [B, 1, H, W]
            masks_cropped = masks[:, :, top:top+new_h, left:left+new_w]
            masks_cropped = F.interpolate(masks_cropped, size=(H, W), mode='nearest')
        else:  # [B, K, H, W]
            masks_cropped = masks[:, :, top:top+new_h, left:left+new_w]
            masks_cropped = F.interpolate(masks_cropped, size=(H, W), mode='nearest')
        
        return imgs_cropped, masks_cropped


class JPEGAttack(BaseAttack):
    """JPEG压缩攻击"""
    def __init__(self, min_quality=30, max_quality=80):
        super().__init__()
        self.min_quality = min_quality
        self.max_quality = max_quality
    
    def forward(self, imgs, masks):
        quality = random.randint(self.min_quality, self.max_quality)
        
        # 对每张图像应用JPEG压缩
        imgs_compressed = []
        for i in range(imgs.shape[0]):
            img = imgs[i]
            # 转换为PIL图像
            img_pil = transforms.ToPILImage()(img.cpu())
            
            # 模拟JPEG压缩
            import io
            buffer = io.BytesIO()
            img_pil.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img_compressed = Image.open(buffer)
            
            # 转换回tensor
            img_tensor = transforms.ToTensor()(img_compressed).to(imgs.device)
            imgs_compressed.append(img_tensor)
        
        imgs_compressed = torch.stack(imgs_compressed)
        
        # JPEG压缩不影响掩码
        return imgs_compressed, masks


class GaussianBlurAttack(BaseAttack):
    """高斯模糊攻击"""
    def __init__(self, min_kernel=3, max_kernel=17):
        super().__init__()
        self.min_kernel = min_kernel
        self.max_kernel = max_kernel
    
    def forward(self, imgs, masks):
        # 确保kernel_size是奇数
        kernel_size = random.randrange(self.min_kernel, self.max_kernel + 1, 2)
        sigma = random.uniform(0.5, 2.0)
        
        # 应用高斯模糊
        imgs_blurred = kornia.filters.gaussian_blur2d(imgs, (kernel_size, kernel_size), (sigma, sigma))
        
        # 模糊不影响掩码
        return imgs_blurred, masks


class BrightnessAttack(BaseAttack):
    """亮度攻击"""
    def __init__(self, min_factor=0.5, max_factor=1.5):
        super().__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor
    
    def forward(self, imgs, masks):
        factor = random.uniform(self.min_factor, self.max_factor)
        imgs_adjusted = torch.clamp(imgs * factor, 0, 1)
        
        # 亮度调整不影响掩码
        return imgs_adjusted, masks


class ContrastAttack(BaseAttack):
    """对比度攻击"""
    def __init__(self, min_factor=0.5, max_factor=1.5):
        super().__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor
    
    def forward(self, imgs, masks):
        factor = random.uniform(self.min_factor, self.max_factor)
        
        # 计算图像均值
        mean_vals = imgs.mean(dim=[2, 3], keepdim=True)
        
        # 调整对比度
        imgs_adjusted = torch.clamp(factor * (imgs - mean_vals) + mean_vals, 0, 1)
        
        # 对比度调整不影响掩码
        return imgs_adjusted, masks


class WatermarkAugmenter(nn.Module):
    """
    水印攻击增强器，模拟各种攻击场景
    """
    
    def __init__(self, attacks_config=None):
        super().__init__()
        
        if attacks_config is None:
            # 默认攻击配置
            attacks_config = {
                'identity': {'weight': 4.0},
                'rotation': {'weight': 1.0, 'min_angle': -10, 'max_angle': 10},
                'resize': {'weight': 1.0, 'min_scale': 0.7, 'max_scale': 1.3},
                'crop': {'weight': 1.0, 'min_crop': 0.7, 'max_crop': 0.9},
                'jpeg': {'weight': 1.0, 'min_quality': 50, 'max_quality': 90},
                'blur': {'weight': 1.0, 'min_kernel': 3, 'max_kernel': 9},
                'brightness': {'weight': 0.5, 'min_factor': 0.8, 'max_factor': 1.2},
                'contrast': {'weight': 0.5, 'min_factor': 0.8, 'max_factor': 1.2},
            }
        
        self.attacks = []
        self.weights = []
        
        # 构建攻击列表
        attack_map = {
            'identity': IdentityAttack,
            'rotation': RotationAttack,
            'resize': ResizeAttack,
            'crop': CropAttack,
            'jpeg': JPEGAttack,
            'blur': GaussianBlurAttack,
            'brightness': BrightnessAttack,
            'contrast': ContrastAttack,
        }
        
        for attack_name, config in attacks_config.items():
            if attack_name in attack_map:
                weight = config.pop('weight', 1.0)
                attack = attack_map[attack_name](**config)
                self.attacks.append(attack)
                self.weights.append(weight)
        
        # 归一化权重
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        self.weights = torch.tensor(self.weights)
    
    def forward(self, imgs, masks):
        """
        随机选择一种攻击并应用
        
        Args:
            imgs: torch.Tensor [B, C, H, W] - 输入图像
            masks: torch.Tensor [B, 1, H, W] or [B, K, H, W] - 掩码
            
        Returns:
            tuple: (attacked_imgs, attacked_masks, attack_name)
        """
        # 随机选择攻击
        attack_idx = torch.multinomial(self.weights, 1).item()
        selected_attack = self.attacks[attack_idx]
        
        # 应用攻击
        attacked_imgs, attacked_masks = selected_attack(imgs, masks)
        
        attack_name = selected_attack.__class__.__name__
        
        return attacked_imgs, attacked_masks, attack_name
    
    def __repr__(self):
        attack_names = [attack.__class__.__name__ for attack in self.attacks]
        return f"WatermarkAugmenter(attacks={attack_names}, weights={self.weights.tolist()})"


# 便于import的函数
def build_augmenter(config=None):
    """构建水印攻击增强器"""
    return WatermarkAugmenter(config)

