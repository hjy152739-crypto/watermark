# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
torchrun --nproc_per_node=2 main.py \
    --local_rank 0 \
    --encoder vit --encoder_depth 12 --encoder_channels 384 --use_tanh True \
    --loss_margin 100 --scaling_w 0.5 \
    --batch_size 16 --eval_freq 10 \
    --attenuation jnd \
    --epochs 100 --optimizer "AdamW,lr=1e-4"
    
Args Inventory:
    --dist False \
    --encoder vit --encoder_depth 6 --encoder_channels 384 --use_tanh True \
    --batch_size 128 --batch_size_eval 128 --workers 8 \
    --attenuation jnd \
    --num_bits 64 --redundancy 16 \
    --encoder vit --encoder_depth 6 --encoder_channels 384 --use_tanh True \
    --encoder vit --encoder_depth 12 --encoder_channels 384 --use_tanh True \
    --loss_margin 100   --attenuation jnd --batch_size 16 --eval_freq 10 --local_rank 0 \
    --p_crop 0 --p_rot 0 --p_color_jitter 0 --p_blur 0 --p_jpeg 0 --p_res 0 \

"""

import argparse
import datetime
import json
import os
import time
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.transforms import functional
from torchvision.utils import save_image

import data_augmentation
import utils
import utils_img
import models
import attenuations

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Experiments parameters')
    aa("--train_dir", type=str, default="coco/train")
    aa("--val_dir", type=str, default="coco/val")
    aa("--train_annotation_file", type=str, default=None, help="COCO train annotation file path")
    aa("--val_annotation_file", type=str, default=None, help="COCO val annotation file path")
    aa("--output_dir", type=str, default="output/", help="Output directory for logs and images (Default: /output)")
    aa("--use_coco_segmentation", type=utils.bool_inst, default=False, help="Use COCO segmentation dataset")
    aa("--use_localized_watermark", type=utils.bool_inst, default=False, help="Use localized watermark training")
    aa("--max_watermarks", type=int, default=4, help="Maximum number of watermarks per image")
    aa("--multi_watermark_prob", type=float, default=0.3, help="Probability of using multiple watermarks")

    group = parser.add_argument_group('Marking parameters')
    aa("--num_bits", type=int, default=32, help="Number of bits of the watermark (Default: 32)")
    aa("--redundancy", type=int, default=1, help="Redundancy of the watermark (Default: 1)")
    aa("--img_size", type=int, default=128, help="Image size")

    group = parser.add_argument_group('Encoder parameters')
    aa("--encoder", type=str, default="hidden", help="Encoder type (Default: hidden)")
    aa('--encoder_depth', default=4, type=int, help='Number of blocks in the encoder.')
    aa('--encoder_channels', default=64, type=int, help='Number of channels in the encoder.')
    aa("--use_tanh", type=utils.bool_inst, default=True, help="Use tanh scaling. (Default: True)")

    group = parser.add_argument_group('Decoder parameters')
    aa("--decoder", type=str, default="hidden", help="Decoder type (Default: hidden, advanced)")
    aa("--decoder_depth", type=int, default=8, help="Number of blocks in the decoder (Default: 4)")
    aa("--decoder_channels", type=int, default=64, help="Number of blocks in the decoder (Default: 4)")
    aa("--decoder_architecture", type=str, default="unet", help="Advanced decoder architecture (unet, vit)")
    aa("--use_advanced_decoder", type=utils.bool_inst, default=False, help="Use advanced decoder with DBSCAN (Default: False)")
    aa("--dbscan_epsilon", type=float, default=1.0, help="DBSCAN epsilon parameter (Default: 1.0)")
    aa("--dbscan_min_samples", type=int, default=100, help="DBSCAN min_samples parameter (Default: 100)")

    group = parser.add_argument_group('Training parameters')
    aa("--bn_momentum", type=float, default=0.01, help="Momentum of the batch normalization layer. (Default: 0.1)")
    aa('--eval_freq', default=1, type=int)
    aa('--saveckp_freq', default=100, type=int)
    aa('--saveimg_freq', default=10, type=int)
    aa('--resume_from', default=None, type=str, help='Checkpoint path to resume from.')
    aa("--scaling_w", type=float, default=1.0, help="Scaling of the watermark signal. (Default: 1.0)")
    aa("--scaling_i", type=float, default=1.0, help="Scaling of the original image. (Default: 1.0)")

    group = parser.add_argument_group('Optimization parameters')
    aa("--epochs", type=int, default=400, help="Number of epochs for optimization. (Default: 100)")
    aa("--optimizer", type=str, default="Adam", help="Optimizer to use. (Default: Adam)")
    aa("--scheduler", type=str, default=None, help="Scheduler to use. (Default: None)")
    aa("--lambda_w", type=float, default=1.0, help="Weight of the watermark loss. (Default: 1.0)")
    aa("--lambda_i", type=float, default=0.0, help="Weight of the image loss. (Default: 0.0)")
    aa("--lambda_det", type=float, default=1.0, help="Weight of the detection loss (λ_det). (Default: 1.0)")
    aa("--lambda_dec", type=float, default=1.0, help="Weight of the decoding loss (λ_dec). (Default: 1.0)")
    aa("--loss_margin", type=float, default=1, help="Margin of the Hinge loss or temperature of the sigmoid of the BCE loss. (Default: 1.0)")
    aa("--loss_i_type", type=str, default='mse', help="Loss type. 'mse' for mean squared error, 'l1' for l1 loss (Default: mse)")
    aa("--loss_w_type", type=str, default='bce', help="Loss type. 'bce' for binary cross entropy, 'cossim' for cosine similarity (Default: bce)")
    aa("--use_pixel_level_loss", type=utils.bool_inst, default=False, help="Use pixel-level detection and decoding loss (Default: False)")

    group = parser.add_argument_group('Loader parameters')
    aa("--batch_size", type=int, default=16, help="Batch size. (Default: 16)")
    aa("--batch_size_eval", type=int, default=64, help="Batch size. (Default: 128)")
    aa("--workers", type=int, default=8, help="Number of workers for data loading. (Default: 8)")

    group = parser.add_argument_group('Attenuation parameters')
    aa("--attenuation", type=str, default=None, help="Attenuation type. (Default: jnd)")
    aa("--scale_channels", type=utils.bool_inst, default=True, help="Use channel scaling. (Default: True)")

    group = parser.add_argument_group('DA parameters')
    aa("--data_augmentation", type=str, default="combined", help="Type of data augmentation to use at marking time. (Default: combined)")
    aa("--p_crop", type=float, default=0.5, help="Probability of the crop augmentation. (Default: 0.5)")
    aa("--p_res", type=float, default=0.5, help="Probability of the crop augmentation. (Default: 0.5)")
    aa("--p_blur", type=float, default=0.5, help="Probability of the blur augmentation. (Default: 0.5)")
    aa("--p_jpeg", type=float, default=0.5, help="Probability of the diff JPEG augmentation. (Default: 0.5)")
    aa("--p_rot", type=float, default=0.5, help="Probability of the rotation augmentation. (Default: 0.5)")
    aa("--p_color_jitter", type=float, default=0.5, help="Probability of the color jitter augmentation. (Default: 0.5)")
    
    group = parser.add_argument_group('Attack simulation parameters')
    aa("--use_advanced_attacks", type=utils.bool_inst, default=False, help="Use advanced attack simulation")
    aa("--attack_identity_weight", type=float, default=4.0, help="Weight for identity attack")
    aa("--attack_rotation_weight", type=float, default=1.0, help="Weight for rotation attack")
    aa("--attack_resize_weight", type=float, default=1.0, help="Weight for resize attack")
    aa("--attack_crop_weight", type=float, default=1.0, help="Weight for crop attack")
    aa("--attack_jpeg_weight", type=float, default=1.0, help="Weight for JPEG attack")
    aa("--attack_blur_weight", type=float, default=1.0, help="Weight for blur attack")
    


    group = parser.add_argument_group('Distributed training parameters')
    aa('--debug_slurm', action='store_true')
    aa('--local_rank', default=-1, type=int)
    aa('--master_port', default=-1, type=int)
    aa('--dist', type=utils.bool_inst, default=True, help='Enabling distributed training')

    group = parser.add_argument_group('Misc')
    aa('--seed', default=0, type=int, help='Random seed')

    return parser



def main(params):
    # Distributed mode
    if params.dist:
        utils.init_distributed_mode(params)
        # cudnn.benchmark = False
        # cudnn.deterministic = True

    # Set seeds for reproductibility 
    seed = params.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Print the arguments
    print("__git__:{}".format(utils.get_sha()))
    print("__log__:{}".format(json.dumps(vars(params))))

    # handle params that are "none"
    if params.attenuation is not None:
        if params.attenuation.lower() == 'none':
            params.attenuation = None
    if params.scheduler is not None:
        if params.scheduler.lower() == 'none':
            params.scheduler = None

    # Build encoder
    print('building encoder...')
    if params.encoder == 'hidden':
        encoder = models.HiddenEncoder(num_blocks=params.encoder_depth, num_bits=params.num_bits, channels=params.encoder_channels, last_tanh=params.use_tanh)
    elif params.encoder == 'dvmark':
        encoder = models.DvmarkEncoder(num_blocks=params.encoder_depth, num_bits=params.num_bits, channels=params.encoder_channels, last_tanh=params.use_tanh)
    elif params.encoder == 'vit':
        encoder = models.VitEncoder(
            img_size=params.img_size, patch_size=16, init_values=None,
            embed_dim=params.encoder_channels, depth=params.encoder_depth, 
            num_bits=params.num_bits, last_tanh=params.use_tanh
            )
    else:
        raise ValueError('Unknown encoder type')
    print('\nencoder: \n%s'% encoder)
    print('total parameters: %d'%sum(p.numel() for p in encoder.parameters()))

    # Build decoder
    print('building decoder...')
    if params.use_advanced_decoder or params.decoder == 'advanced':
        decoder = models.AdvancedHiddenDecoder(
            num_bits=params.num_bits*params.redundancy, 
            channels=params.decoder_channels,
            img_size=params.img_size,
            architecture=params.decoder_architecture,
            use_advanced=True,
            dbscan_epsilon=params.dbscan_epsilon,
            dbscan_min_samples=params.dbscan_min_samples
        )
    elif params.decoder == 'hidden':
        decoder = models.HiddenDecoder(num_blocks=params.decoder_depth, num_bits=params.num_bits*params.redundancy, channels=params.decoder_channels)
    else:
        raise ValueError('Unknown decoder type')
    print('\ndecoder: \n%s'% decoder)
    print('total parameters: %d'%sum(p.numel() for p in decoder.parameters()))
    
    # Adapt bn momentum
    for module in [*decoder.modules(), *encoder.modules()]:
        if type(module) == torch.nn.BatchNorm2d:
            module.momentum = params.bn_momentum if params.bn_momentum != -1 else None

    # Construct attenuation
    if params.attenuation == 'jnd':
        attenuation = attenuations.JND(preprocess = lambda x: utils_img.unnormalize_rgb(x)).to(device)
    else:
        attenuation = None

    # Construct data augmentation seen at train time
    if params.data_augmentation == 'combined':
        data_aug = data_augmentation.HiddenAug(params.img_size, params.p_crop, params.p_blur,  params.p_jpeg, params.p_rot,  params.p_color_jitter, params.p_res).to(device)
    elif params.data_augmentation == 'kornia':
        data_aug = data_augmentation.KorniaAug().to(device)
    elif params.data_augmentation == 'none':
        data_aug = nn.Identity().to(device)
    else:
        raise ValueError('Unknown data augmentation type')
    print('data augmentation: %s'%data_aug)
    
    # Create encoder/decoder
    if params.use_localized_watermark:
        print("Using LocalizedEncoderDecoder with advanced features...")
        
        # 构建攻击配置
        attack_config = None
        if params.use_advanced_attacks:
            attack_config = {
                'identity': {'weight': params.attack_identity_weight},
                'rotation': {'weight': params.attack_rotation_weight, 'min_angle': -10, 'max_angle': 10},
                'resize': {'weight': params.attack_resize_weight, 'min_scale': 0.7, 'max_scale': 1.3},
                'crop': {'weight': params.attack_crop_weight, 'min_crop': 0.7, 'max_crop': 0.9},
                'jpeg': {'weight': params.attack_jpeg_weight, 'min_quality': 50, 'max_quality': 90},
                'blur': {'weight': params.attack_blur_weight, 'min_kernel': 3, 'max_kernel': 9},
            }
        
        encoder_decoder = models.LocalizedEncoderDecoder(
            encoder=encoder, 
            attenuation=attenuation, 
            augmentation=data_aug, 
            decoder=decoder,
            scale_channels=params.scale_channels, 
            scaling_i=params.scaling_i, 
            scaling_w=params.scaling_w, 
            num_bits=params.num_bits, 
            redundancy=params.redundancy,
            max_watermarks=params.max_watermarks,
            use_localized=params.use_coco_segmentation,
            use_advanced_attacks=params.use_advanced_attacks,
            attack_config=attack_config
        )
    else:
        print("Using standard EncoderDecoder...")
        encoder_decoder = models.EncoderDecoder(encoder, attenuation, data_aug, decoder, 
            params.scale_channels, params.scaling_i, params.scaling_w, params.num_bits, params.redundancy)
    
    encoder_decoder = encoder_decoder.to(device)

    # Distributed training
    if params.dist:
        encoder_decoder = nn.SyncBatchNorm.convert_sync_batchnorm(encoder_decoder)
        encoder_decoder = nn.parallel.DistributedDataParallel(encoder_decoder, device_ids=[params.local_rank])

    # Build optimizer and scheduler
    optim_params = utils.parse_params(params.optimizer)
    lr_mult = params.batch_size * utils.get_world_size() / 512.0
    optim_params['lr'] = lr_mult * optim_params['lr'] if 'lr' in optim_params else lr_mult * 1e-3
    to_optim = [*encoder.parameters(), *decoder.parameters()]
    optimizer = utils.build_optimizer(model_params=to_optim, **optim_params)
    scheduler = utils.build_lr_scheduler(optimizer=optimizer, **utils.parse_params(params.scheduler)) if params.scheduler is not None else None
    print('optimizer: %s'%optimizer)
    print('scheduler: %s'%scheduler)

    # Data loaders
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(params.img_size),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        utils_img.normalize_rgb,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(params.img_size),
        transforms.CenterCrop(params.img_size),
        transforms.ToTensor(),
        utils_img.normalize_rgb,
        ])
    
    # 选择数据加载器类型
    if params.use_coco_segmentation and params.train_annotation_file and params.val_annotation_file:
        print("Using COCO segmentation dataset...")
        from coco_segmentation_loader import get_coco_segmentation_loader
        
        # 掩码变换
        mask_transform = transforms.Compose([
            transforms.Resize(params.img_size),
            transforms.CenterCrop(params.img_size),
        ])
        
        train_loader = get_coco_segmentation_loader(
            data_dir=params.train_dir,
            ann_file=params.train_annotation_file,
            transform=train_transform,
            mask_transform=mask_transform,
            batch_size=params.batch_size,
            num_workers=params.workers,
            shuffle=True,
            multi_w=(params.multi_watermark_prob > 0),
            max_nb_masks=params.max_watermarks
        )
        
        val_loader = get_coco_segmentation_loader(
            data_dir=params.val_dir,
            ann_file=params.val_annotation_file,
            transform=val_transform,
            mask_transform=mask_transform,
            batch_size=params.batch_size_eval,
            num_workers=params.workers,
            shuffle=False,
            multi_w=False,  # 验证时不使用多水印
            max_nb_masks=1
        )
    else:
        print("Using standard image dataset...")
        train_loader = utils.get_dataloader(params.train_dir, transform=train_transform, batch_size=params.batch_size, num_workers=params.workers, shuffle=True)
        val_loader = utils.get_dataloader(params.val_dir, transform=val_transform, batch_size=params.batch_size_eval, num_workers=params.workers, shuffle=False)

    # optionally resume training 
    if params.resume_from is not None: 
        utils.restart_from_checkpoint(
            params.resume_from,
            encoder_decoder=encoder_decoder
        )
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(params.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        encoder_decoder=encoder_decoder,
        optimizer=optimizer
    )
    start_epoch = to_restore["epoch"]
    for param_group in optimizer.param_groups:
        param_group['lr'] = optim_params['lr']

    # create output dir
    os.makedirs(params.output_dir, exist_ok=True)

    print('training...')
    start_time = time.time()
    best_bit_acc = 0
    for epoch in range(start_epoch, params.epochs):
        
        if params.dist:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(encoder_decoder, train_loader, optimizer, scheduler, epoch, params)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        
        if epoch % params.eval_freq == 0:
            val_stats = eval_one_epoch(encoder_decoder, val_loader, epoch, params)
            log_stats = {**log_stats, **{f'val_{k}': v for k, v in val_stats.items()}}
    
        save_dict = {
            'encoder_decoder': encoder_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'params': params,
        }
        utils.save_on_master(save_dict, os.path.join(params.output_dir, 'checkpoint.pth'))
        if params.saveckp_freq and epoch % params.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(params.output_dir, f'checkpoint{epoch:03}.pth'))
        if utils.is_main_process():
            with (Path(params.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def message_loss(fts, targets, m, loss_type='mse'):
    """
    Compute the message loss
    Args:
        dot products (b k*r): the dot products between the carriers and the feature
        targets (KxD): boolean message vectors or gaussian vectors
        m: margin of the Hinge loss or temperature of the sigmoid of the BCE loss
    """
    if loss_type == 'bce':
        return F.binary_cross_entropy(torch.sigmoid(fts/m), 0.5*(targets+1), reduction='mean')
    elif loss_type == 'cossim':
        return -torch.mean(torch.cosine_similarity(fts, targets, dim=-1))
    elif loss_type == 'mse':
        return F.mse_loss(fts, targets, reduction='mean')
    else:
        raise ValueError('Unknown loss type')
    
def image_loss(imgs, imgs_ori, loss_type='mse'):
    """
    Compute the image loss
    Args:
        imgs (BxCxHxW): the reconstructed images
        imgs_ori (BxCxHxW): the original images
        loss_type: the type of loss
    """
    if loss_type == 'mse':
        return F.mse_loss(imgs, imgs_ori, reduction='mean')
    if loss_type == 'l1':
        return F.l1_loss(imgs, imgs_ori, reduction='mean')
    else:
        raise ValueError('Unknown loss type')

def detection_loss(pred_mask, gt_mask_final):
    """
    计算检测损失 (L_detection): 预测掩码与最终真值掩码之间的逐像素二元交叉熵损失
    Args:
        pred_mask (Bx1xHxW): 预测的水印掩码
        gt_mask_final (Bx1xHxW): 经过攻击变换的最终真值掩码
    Returns:
        detection_loss: 逐像素二元交叉熵损失
    """
    return F.binary_cross_entropy_with_logits(pred_mask, gt_mask_final.float(), reduction='mean')

def decoding_loss(pred_msg_tensor, gt_msg_tensor, gt_mask_final):
    """
    计算解码损失 (L_decoding): 只在gt_mask_final为1的区域，计算pred_msg_tensor与gt_msg_tensor之间的逐像素二元交叉熵损失
    Args:
        pred_msg_tensor (BxKxHxW): 预测的逐像素消息
        gt_msg_tensor (BxK): 真值消息
        gt_mask_final (Bx1xHxW): 经过攻击变换的最终真值掩码
    Returns:
        decoding_loss: 掩码区域内的逐像素消息二元交叉熵损失
    """
    B, K, H, W = pred_msg_tensor.shape
    
    # 将真值消息扩展到空间维度 [B, K] -> [B, K, H, W]
    gt_msg_expanded = gt_msg_tensor.unsqueeze(-1).unsqueeze(-1).expand(B, K, H, W)
    
    # 将掩码扩展到消息维度 [B, 1, H, W] -> [B, K, H, W]
    mask_expanded = gt_mask_final.expand(B, K, H, W)
    
    # 只在掩码区域计算损失
    valid_pixels = mask_expanded > 0.5  # 布尔掩码
    
    if valid_pixels.sum() == 0:
        # 如果没有有效像素，返回零损失
        return torch.tensor(0.0, device=pred_msg_tensor.device, requires_grad=True)
    
    # 提取掩码区域内的预测和真值
    pred_valid = pred_msg_tensor[valid_pixels]  # [num_valid_pixels]
    gt_valid = gt_msg_expanded[valid_pixels]    # [num_valid_pixels]
    
    # 转换真值消息从 {-1, 1} 到 {0, 1}
    gt_valid_01 = (gt_valid + 1) / 2
    
    # 计算二元交叉熵损失
    return F.binary_cross_entropy_with_logits(pred_valid, gt_valid_01, reduction='mean')

def train_one_epoch(encoder_decoder, loader, optimizer, scheduler, epoch, params):
    """
    One epoch of training - 支持局部化水印和多水印训练
    """
    if params.scheduler is not None:
        scheduler.step(epoch)
    encoder_decoder.train()
    header = 'Train - Epoch: [{}/{}]'.format(epoch, params.epochs)
    metric_logger = utils.MetricLogger(delimiter="  ")

    for it, batch_data in enumerate(metric_logger.log_every(loader, 10, header)):
        # 处理不同的数据格式
        if len(batch_data) == 2:
            imgs, masks_or_labels = batch_data
            if params.use_coco_segmentation:
                # COCO分割数据：imgs和masks
                imgs = imgs.to(device, non_blocking=True)  # b c h w
                gt_masks = masks_or_labels.to(device, non_blocking=True)  # b k h w 或 b 1 h w
            else:
                # 标准图像数据：imgs和labels（忽略labels）
                imgs = imgs.to(device, non_blocking=True)  # b c h w
                gt_masks = None
        else:
            # 向后兼容
            imgs = batch_data[0].to(device, non_blocking=True)
            gt_masks = None

        # 生成随机消息
        msgs_ori = torch.rand((imgs.shape[0], params.num_bits)) > 0.5  # b k
        msgs = 2 * msgs_ori.type(torch.float).to(device) - 1  # b k

        # 前向传播，自动支持多水印嵌入
        if isinstance(encoder_decoder, models.LocalizedEncoderDecoder):
            # 使用支持多水印的LocalizedEncoderDecoder
            if params.use_advanced_decoder and isinstance(encoder_decoder.decoder, models.AdvancedHiddenDecoder):
                # 高级解码器：返回详细结果
                outputs = encoder_decoder(imgs, msgs, gt_masks, eval_mode=False, return_detailed=True)
                fts = outputs['fts']
                imgs_w = outputs['imgs_w_local']
                imgs_aug = outputs['imgs_aug']
                
                # 获取预测结果
                mask_pred = outputs.get('mask_pred', None)
                bit_pred = outputs.get('bit_pred', None)
                gt_mask_final = outputs.get('gt_mask_final', gt_masks)
                
                # 获取多水印统计信息
                num_watermarks = outputs.get('num_watermarks', 1)
                msgs_list = outputs.get('msgs_list', [msgs])
                
            else:
                # 传统解码器：返回传统格式
                fts, (imgs_w, imgs_aug) = encoder_decoder(imgs, msgs, gt_masks, eval_mode=False)
                mask_pred = None
                bit_pred = None
                gt_mask_final = gt_masks
                num_watermarks = 1
                msgs_list = [msgs]
        else:
            # 使用原始编码器-解码器
            fts, (imgs_w, imgs_aug) = encoder_decoder(imgs, msgs)
            mask_pred = None
            bit_pred = None
            gt_mask_final = gt_masks
            num_watermarks = 1
            msgs_list = [msgs]

        # 计算损失（仅两项：水印损失与掩码逐像素损失），不计算图像重建损失
        if params.use_pixel_level_loss and mask_pred is not None and bit_pred is not None and gt_mask_final is not None:
            # 处理多掩码情况：合并为单一掩码
            if len(gt_mask_final.shape) == 4 and gt_mask_final.shape[1] > 1:
                gt_mask_combined = torch.clamp(torch.sum(gt_mask_final, dim=1, keepdim=True), 0, 1)
            else:
                gt_mask_combined = gt_mask_final
            
            # 掩码逐像素损失（检测）
            loss_mask = detection_loss(mask_pred, gt_mask_combined)
            # 水印损失（逐像素解码）
            loss_wm = decoding_loss(bit_pred, msgs, gt_mask_combined)
            
            # 总损失
            loss = params.lambda_det * loss_mask + params.lambda_dec * loss_wm
        else:
            # 传统路径：仅使用消息损失 +（可选）掩码分割损失
            loss_wm = message_loss(fts, msgs, m=params.loss_margin, loss_type=params.loss_w_type)
            # 掩码逐像素损失（如果有预测掩码和真值掩码）
            if mask_pred is not None and gt_mask_final is not None:
                if len(gt_mask_final.shape) == 4 and gt_mask_final.shape[1] > 1:
                    gt_mask_combined = torch.clamp(torch.sum(gt_mask_final, dim=1, keepdim=True), 0, 1)
                else:
                    gt_mask_combined = gt_mask_final
                loss_mask = F.binary_cross_entropy_with_logits(mask_pred, gt_mask_combined.float())
            else:
                loss_mask = torch.tensor(0.0, device=imgs.device)
            # 总损失
            loss = params.lambda_w * loss_wm + params.lambda_det * loss_mask

        # 梯度更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计信息
        psnrs = utils_img.psnr(imgs_w, imgs)  # b 1
        # 消息统计
        ori_msgs = torch.sign(msgs) > 0
        decoded_msgs = torch.sign(fts) > 0  # b k -> b k
        diff = (~torch.logical_xor(ori_msgs, decoded_msgs))  # b k -> b k
        bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]  # b k -> b
        word_accs = (bit_accs == 1)  # b
        norm = torch.norm(fts, dim=-1, keepdim=True)  # b d -> b 1
        
        log_stats = {
            'loss_wm': loss_wm.item(),
            'loss_mask': loss_mask.item() if isinstance(loss_mask, torch.Tensor) else float(loss_mask),
            'loss': loss.item(),
            'psnr_avg': torch.mean(psnrs).item(),
            'lr': optimizer.param_groups[0]['lr'],
            'bit_acc_avg': torch.mean(bit_accs).item(),
            'word_acc_avg': torch.mean(word_accs.type(torch.float)).item(),
            'norm_avg': torch.mean(norm).item(),
        }
        
        # 记录权重系数
        if params.use_pixel_level_loss and mask_pred is not None and bit_pred is not None:
            log_stats['lambda_det'] = params.lambda_det
            log_stats['lambda_dec'] = params.lambda_dec
        
        # 添加多水印训练统计
        log_stats['num_watermarks'] = num_watermarks
        if num_watermarks > 1:
            log_stats['is_multi_watermark'] = 1.0
        else:
            log_stats['is_multi_watermark'] = 0.0
        
        torch.cuda.synchronize()
        for name, loss in log_stats.items():
            metric_logger.update(**{name:loss})
        
        # 保存训练可视化：原图、GT掩码、水印图、预测掩码
        if epoch % params.saveimg_freq == 0 and it == 0 and utils.is_main_process():
            # 原始图像
            save_image(utils_img.unnormalize_img(imgs), os.path.join(params.output_dir, f'{epoch:03}_{it:03}_train_ori.png'), nrow=8)
            # 水印图像
            save_image(utils_img.unnormalize_img(imgs_w), os.path.join(params.output_dir, f'{epoch:03}_{it:03}_train_w.png'), nrow=8)

            # 生成并保存GT掩码与预测掩码（若存在）
            gt_mask_vis = None
            pred_mask_vis = None
            try:
                if gt_mask_final is not None:
                    if len(gt_mask_final.shape) == 4 and gt_mask_final.shape[1] > 1:
                        gt_mask_combined_vis = torch.clamp(torch.sum(gt_mask_final, dim=1, keepdim=True), 0, 1)
                    else:
                        gt_mask_combined_vis = gt_mask_final
                    gt_mask_vis = gt_mask_combined_vis.float().repeat(1, 3, 1, 1)  # 转为3通道以便保存
                if mask_pred is not None:
                    pred_mask_prob = torch.sigmoid(mask_pred)
                    pred_mask_vis = pred_mask_prob.repeat(1, 3, 1, 1)
            except Exception:
                gt_mask_vis = None
                pred_mask_vis = None

            if gt_mask_vis is not None:
                save_image(gt_mask_vis, os.path.join(params.output_dir, f'{epoch:03}_{it:03}_train_mask_gt.png'), nrow=8)
            if pred_mask_vis is not None:
                save_image(pred_mask_vis, os.path.join(params.output_dir, f'{epoch:03}_{it:03}_train_mask_pred.png'), nrow=8)

    metric_logger.synchronize_between_processes()
    print("Averaged {} stats:".format('train'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def eval_one_epoch(encoder_decoder: models.EncoderDecoder, loader, epoch, params):
    """
    One epoch of eval.
    """
    encoder_decoder.eval()
    header = 'Eval - Epoch: [{}/{}]'.format(epoch, params.epochs)
    metric_logger = utils.MetricLogger(delimiter="  ")
    for it, (imgs, _) in enumerate(metric_logger.log_every(loader, 10, header)):
        imgs = imgs.to(device, non_blocking=True) # b c h w

        msgs_ori = torch.rand((imgs.shape[0],params.num_bits)) > 0.5 # b k
        msgs = 2 * msgs_ori.type(torch.float).to(device) - 1 # b k

        fts, (imgs_w, imgs_aug) = encoder_decoder(imgs, msgs, eval_mode=True)
        
        loss_w = message_loss(fts, msgs,  m=params.loss_margin, loss_type=params.loss_w_type) # b -> 1
        loss_i = image_loss(imgs_w, imgs, loss_type=params.loss_i_type) # b c h w -> 1
        
        loss = params.lambda_w*loss_w + params.lambda_i*loss_i

        # img stats
        psnrs = utils_img.psnr(imgs_w, imgs) # b 1
        # msg stats
        ori_msgs = torch.sign(msgs) > 0
        decoded_msgs = torch.sign(fts) > 0 # b k -> b k
        diff = (~torch.logical_xor(ori_msgs, decoded_msgs)) # b k -> b k
        bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
        word_accs = (bit_accs == 1) # b
        norm = torch.norm(fts, dim=-1, keepdim=True) # b d -> b 1
        log_stats = {
            'loss_w': loss_w.item(),
            'loss_i': loss_i.item(),
            'loss': loss.item(),
            'psnr_avg': torch.mean(psnrs).item(),
            'bit_acc_avg': torch.mean(bit_accs).item(),
            'word_acc_avg': torch.mean(word_accs.type(torch.float)).item(),
            'norm_avg': torch.mean(norm).item(),
        }

        attacks = {
            'none': lambda x: x,
            'crop_01': lambda x: utils_img.center_crop(x, 0.1),
            'crop_05': lambda x: utils_img.center_crop(x, 0.5),
            # 'resize_03': lambda x: utils_img.resize(x, 0.3),
            'resize_05': lambda x: utils_img.resize(x, 0.5),
            'rot_25': lambda x: utils_img.rotate(x, 25),
            'rot_90': lambda x: utils_img.rotate(x, 90),
            'blur': lambda x: utils_img.gaussian_blur(x, sigma=2.0),
            # 'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
            'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
        }
        for name, attack in attacks.items():
            fts, (_) = encoder_decoder(imgs, msgs, eval_mode=True, eval_aug=attack)
            decoded_msgs = torch.sign(fts) > 0 # b k -> b k
            diff = (~torch.logical_xor(ori_msgs, decoded_msgs)) # b k -> b k
            log_stats[f'bit_acc_{name}'] = diff.float().mean().item()

        torch.cuda.synchronize()
        for name, loss in log_stats.items():
            metric_logger.update(**{name:loss})
        
        if epoch % params.saveimg_freq == 0 and it == 0 and utils.is_main_process():
            # 原始图像与水印图像
            save_image(utils_img.unnormalize_img(imgs), os.path.join(params.output_dir, f'{epoch:03}_{it:03}_val_ori.png'), nrow=8)
            save_image(utils_img.unnormalize_img(imgs_w), os.path.join(params.output_dir, f'{epoch:03}_{it:03}_val_w.png'), nrow=8)
            # GT掩码（若有）与预测掩码（若有）
            gt_mask_vis = None
            pred_mask_vis = None
            try:
                # 评估阶段没有增强返回的mask，只能展示合并的训练掩码或空
                gt_mask = None
            except Exception:
                gt_mask = None
            if gt_mask is not None:
                if len(gt_mask.shape) == 4 and gt_mask.shape[1] > 1:
                    gt_mask_combined_vis = torch.clamp(torch.sum(gt_mask, dim=1, keepdim=True), 0, 1)
                else:
                    gt_mask_combined_vis = gt_mask
                gt_mask_vis = gt_mask_combined_vis.float().repeat(1, 3, 1, 1)
            # 预测掩码在eval里不可用，这里仅在未来扩展时保留占位
            if gt_mask_vis is not None:
                save_image(gt_mask_vis, os.path.join(params.output_dir, f'{epoch:03}_{it:03}_val_mask_gt.png'), nrow=8)

    metric_logger.synchronize_between_processes()
    print("Averaged {} stats:".format('eval'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)
