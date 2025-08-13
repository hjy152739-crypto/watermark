# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
COCO分割数据集加载器，用于支持语义分割掩码的局部化水印训练
"""

import os
import functools
import numpy as np
from pycocotools import mask as maskUtils
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import CocoDetection
from torchvision.datasets.folder import is_image_file, default_loader
from torchvision.transforms import Compose, ToTensor, Normalize

import utils


@functools.lru_cache()
def get_image_paths(path):
    paths = []
    for path, _, files in os.walk(path):
        for filename in files:
            paths.append(os.path.join(path, filename))
    return sorted([fn for fn in paths if is_image_file(fn)])


class CocoSegmentationWrapper(CocoDetection):
    """COCO分割数据集包装器，支持多水印和掩码变换"""
    
    def __init__(self, root, annFile, transform=None, mask_transform=None, 
                 random_nb_object=True, max_nb_masks=4, multi_w=False):
        super().__init__(root, annFile, transform=transform, target_transform=mask_transform)
        self.random_nb_object = random_nb_object
        self.max_nb_masks = max_nb_masks
        self.multi_w = multi_w
        self.mask_transform = mask_transform

    def __getitem__(self, index: int):
        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")

        id = self.ids[index]
        img = self._load_image(id)
        mask = self._load_mask(id)
        
        if mask is None:
            return None  # Skip this image if no valid mask is available

        # Apply transforms
        if self.transform:
            img = self.transform(img)
        if self.mask_transform and not self.multi_w:
            mask = self.mask_transform(mask)
            
        return img, mask

    def _load_mask(self, id):
        """加载分割掩码"""
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        if not anns:
            return None  # Return None if there are no annotations

        img_info = self.coco.loadImgs(id)[0]
        original_height = img_info['height']
        original_width = img_info['width']

        # 随机选择对象数量
        if self.random_nb_object and np.random.rand() < 0.5:
            random.shuffle(anns)
            anns = anns[:np.random.randint(1, len(anns)+1)]

        if not self.multi_w:
            # 单一掩码：合并所有对象
            mask = np.zeros((original_height, original_width), dtype=np.float32)
            for ann in anns:
                rle = self.coco.annToRLE(ann)
                m = maskUtils.decode(rle)
                mask = np.maximum(mask, m)
            mask = torch.tensor(mask, dtype=torch.float32)
            return mask.unsqueeze(0)  # Add channel dimension
        else:
            # 多水印掩码：每个对象一个掩码
            anns = anns[:self.max_nb_masks]
            masks = []
            for ann in anns:
                rle = self.coco.annToRLE(ann)
                m = maskUtils.decode(rle)
                masks.append(m)
            
            if masks:
                masks = np.stack(masks, axis=0)
                masks = torch.tensor(masks, dtype=torch.float32)
                # 填充到max_nb_masks
                if masks.shape[0] < self.max_nb_masks:
                    additional_masks_count = self.max_nb_masks - masks.shape[0]
                    additional_masks = torch.zeros(
                        (additional_masks_count, original_height, original_width), 
                        dtype=torch.float32
                    )
                    masks = torch.cat([masks, additional_masks], dim=0)
            else:
                # 如果没有掩码，返回全零掩码
                masks = torch.zeros(
                    (self.max_nb_masks, original_height, original_width), 
                    dtype=torch.float32
                )
            return masks


def custom_collate(batch: list):
    """自定义collate函数，处理None值和掩码对齐"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])
    
    images, masks = zip(*batch)
    images = torch.stack(images)
    # 目标尺寸取自图像张量
    target_h, target_w = images.shape[-2], images.shape[-1]
    
    # 处理掩码维度对齐
    if len(masks[0].shape) == 3:  # 多水印情况
        max_masks = max(mask.shape[0] for mask in masks)
        resized_padded_masks = []
        for mask in masks:
            # 调整空间尺寸到与图像一致
            if mask.shape[-2:] != (target_h, target_w):
                mask = F.interpolate(mask.unsqueeze(0), size=(target_h, target_w), mode='nearest').squeeze(0)
            # 对通道数（掩码数量）进行填充
            if mask.shape[0] < max_masks:
                pad_c = max_masks - mask.shape[0]
                pad_tensor = torch.zeros((pad_c, target_h, target_w), dtype=mask.dtype, device=mask.device)
                mask = torch.cat([mask, pad_tensor], dim=0)
            resized_padded_masks.append(mask)
        masks = torch.stack(resized_padded_masks)
    else:  # 单一掩码情况
        resized_masks = []
        for mask in masks:
            if mask.shape[-2:] != (target_h, target_w):
                mask = F.interpolate(mask.unsqueeze(0), size=(target_h, target_w), mode='nearest').squeeze(0)
            resized_masks.append(mask)
        masks = torch.stack(resized_masks)
    
    return images, masks


def get_coco_segmentation_loader(
    data_dir: str, 
    ann_file: str,
    transform: callable,
    mask_transform: callable = None,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 8,
    random_nb_object: bool = True,
    multi_w: bool = False,
    max_nb_masks: int = 4
) -> DataLoader:
    """获取COCO分割数据集的DataLoader"""
    
    dataset = CocoSegmentationWrapper(
        root=data_dir, 
        annFile=ann_file, 
        transform=transform, 
        mask_transform=mask_transform,
        random_nb_object=random_nb_object, 
        multi_w=multi_w, 
        max_nb_masks=max_nb_masks
    )

    if utils.is_dist_avail_and_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=sampler, 
            num_workers=num_workers, 
            pin_memory=True, 
            drop_last=True, 
            collate_fn=custom_collate
        )
    else:
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers, 
            pin_memory=True, 
            drop_last=True, 
            collate_fn=custom_collate
        )
    
    return dataloader
