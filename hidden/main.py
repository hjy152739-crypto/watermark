# main.py

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
import models # 确保这里的models.py是你刚刚修改过的
import attenuations
from augmentation_attacks import build_augmenter
from coco_segmentation_loader import get_coco_segmentation_loader

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ... (保留 get_parser() 函数, 无需修改) ...
def get_parser():
    # ... (此处省略，保持原样)
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Experiments parameters')
    aa("--train_dir", type=str, default="coco/train")
    aa("--val_dir", type=str, default="coco/val")
    aa("--output_dir", type=str, default="output/", help="Output directory for logs and images (Default: /output)")

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
    aa("--decoder", type=str, default="hidden", help="Decoder type (Default: hidden)")
    aa("--decoder_depth", type=int, default=8, help="Number of blocks in the decoder (Default: 4)")
    aa("--decoder_channels", type=int, default=64, help="Number of blocks in the decoder (Default: 4)")

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
    # NEW: 添加了检测和解码损失的权重
    aa("--lambda_det", type=float, default=1.0, help="Weight of the detection loss. (Default: 1.0)")
    aa("--lambda_dec", type=float, default=1.0, help="Weight of the decoding loss. (Default: 1.0)")
    aa("--lambda_i", type=float, default=0.0, help="Weight of the image loss. (Default: 0.0)")
    
    # ... (保留 get_parser() 的剩余部分) ...
    aa("--loss_margin", type=float, default=1, help="Margin of the Hinge loss or temperature of the sigmoid of the BCE loss. (Default: 1.0)")
    aa("--loss_i_type", type=str, default='mse', help="Loss type. 'mse' for mean squared error, 'l1' for l1 loss (Default: mse)")
    # 这两个参数在我们的新损失函数中不再直接使用，但保留以防万一
    # aa("--loss_w_type", type=str, default='bce', help="Loss type. 'bce' for binary cross entropy, 'cossim' for cosine similarity (Default: bce)")

    group = parser.add_argument_group('Loader parameters')
    aa("--batch_size", type=int, default=16, help="Batch size. (Default: 16)")
    aa("--batch_size_eval", type=int, default=64, help="Batch size. (Default: 128)")
    aa("--workers", type=int, default=8, help="Number of workers for data loading. (Default: 8)")
    aa("--use_coco_segmentation", type=utils.bool_inst, default=False, help="Use COCO segmentation masks for localized training")
    aa("--train_annotation_file", type=str, default=None, help="Path to COCO train annotation json")
    aa("--val_annotation_file", type=str, default=None, help="Path to COCO val annotation json")
    aa("--max_nb_masks", type=int, default=4, help="Max number of masks per image when using COCO segmentation")
    aa("--max_train_images_per_epoch", type=int, default=0, help="Limit number of images per epoch for training (<=0 to disable)")  # 0表示使用完整数据集

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

    group = parser.add_argument_group('Distributed training parameters')
    aa('--debug_slurm', action='store_true')
    aa('--local_rank', default=-1, type=int)
    aa('--master_port', default=-1, type=int)
    aa('--dist', type=utils.bool_inst, default=True, help='Enabling distributed training')

    group = parser.add_argument_group('Misc')
    aa('--seed', default=0, type=int, help='Random seed')

    return parser


# NEW: 生成随机矩形掩码的辅助函数
def create_random_mask(size, batch_size):
    H, W = size
    mask = torch.zeros((batch_size, 1, H, W), device=device)
    for i in range(batch_size):
        # 随机决定掩码大小和位置
        h, w = random.randint(H // 4, H // 2), random.randint(W // 4, W // 2)
        top, left = random.randint(0, H - h), random.randint(0, W - w)
        mask[i, :, top:top+h, left:left+w] = 1
    return mask

# NEW: 计算检测损失 (IoU)
def detection_loss_iou(pred_mask, gt_mask):
    intersection = (pred_mask * gt_mask).sum(dim=[1,2,3])
    union = (pred_mask + gt_mask).sum(dim=[1,2,3]) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return 1 - iou.mean() # 我们希望最大化IoU, 所以最小化 1-IoU

# NEW: 计算解码损失 - 使用预测掩码加权
def decoding_loss(pred_msg_tensor, gt_msg_tensor, pred_mask, gt_mask, use_pred_mask=True):
    """
    计算消息解码损失
    Args:
        pred_msg_tensor: 预测的消息logits [B, num_bits, H, W]
        gt_msg_tensor: 真实消息 [B, num_bits, H, W] 
        pred_mask: 预测的掩码概率 [B, 1, H, W]
        gt_mask: 真实掩码 [B, 1, H, W]
        use_pred_mask: 是否使用预测掩码进行加权
    """
    # 基础BCE损失
    loss = F.binary_cross_entropy_with_logits(pred_msg_tensor, gt_msg_tensor, reduction='none')
    
    if use_pred_mask:
        # 使用预测掩码概率加权：强制模型在预测为水印区域的地方解码正确
        # 同时在真实水印区域给予监督
        mask_weight = pred_mask * gt_mask  # 只在真实水印区域且模型认为是水印的地方计算损失
        total_weight = mask_weight.sum() + 1e-6
    else:
        # 传统方法：只在真实掩码区域计算
        mask_weight = gt_mask  
        total_weight = gt_mask.sum() * gt_msg_tensor.shape[1] + 1e-6
    
    loss = (loss * mask_weight).sum() / total_weight
    return loss

# CHANGED: 整个 train_one_epoch 函数被重写以适应新流程
def train_one_epoch(encoder_decoder: models.EncoderDecoder, loader, optimizer, scheduler, epoch, params, data_aug, water_attacker):
    if params.scheduler is not None:
        scheduler.step(epoch)
    encoder_decoder.train()
    header = 'Train - Epoch: [{}/{}]'.format(epoch, params.epochs)
    metric_logger = utils.MetricLogger(delimiter="  ")

    # 兼容 DDP 与非DDP 的访问
    module = encoder_decoder.module if hasattr(encoder_decoder, 'module') else encoder_decoder
    enc = module['encoder']
    dec = module['decoder']

    # 存储每轮的示例图像用于最终保存
    epoch_images = {
        'original': [],
        'gt_mask': [],
        'watermarked': [],
        'pred_mask': []
    }
    
    for it, batch in enumerate(metric_logger.log_every(loader, 10, header)):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            imgs, batch_masks = batch
        else:
            imgs, batch_masks = batch, None
        imgs = imgs.to(device, non_blocking=True)
        B, C, H, W = imgs.shape

        # 1. 获取/生成掩码与消息
        if batch_masks is not None:
            gt_mask = batch_masks.to(device)
            # 统一为 [B, K, H, W]
            if len(gt_mask.shape) == 3:
                gt_mask = gt_mask.unsqueeze(1)
            K = gt_mask.shape[1]
            # 多掩码：每个掩码区域一条32bit消息
            msgs_list = []
            gt_msg_tensor = torch.zeros((B, params.num_bits, H, W), device=device)
            combined_mask = torch.zeros((B, 1, H, W), device=device)
            # 生成一条统一的消息用于编码器
            msgs_unified_bool = (torch.rand((B, params.num_bits), device=device) > 0.5)
            msgs_unified = msgs_unified_bool.float()  # [B, num_bits], [0,1]
            
            for k in range(K):
                mask_k = gt_mask[:, k:k+1, :, :]
                if mask_k.sum() == 0:
                    continue
                msgs_list.append(msgs_unified)
                # 将统一消息写入该掩码区域
                msgs_k_pix = msgs_unified.view(B, params.num_bits, 1, 1)
                gt_msg_tensor = gt_msg_tensor * (1 - mask_k) + msgs_k_pix * mask_k
                combined_mask = torch.clamp(combined_mask + mask_k, 0, 1)
        else:
            # 随机矩形掩码 + 单条消息
            gt_mask = create_random_mask((H, W), B)
            msgs_unified_bool = (torch.rand((B, params.num_bits), device=device) > 0.5)
            msgs_unified = msgs_unified_bool.float()  # [B, num_bits], [0,1]
            gt_msg_tensor = msgs_unified.view(B, params.num_bits, 1, 1).expand(-1, -1, H, W)
            combined_mask = gt_mask
        
        # 2. 统一水印嵌入（使用统一消息）
        # 将[0,1]消息转换为[-1,1]用于编码器
        msgs_pm1 = 2 * msgs_unified - 1  # [0,1] -> [-1,1]
        watermark_signal = enc(imgs, msgs_pm1)
        imgs_w = imgs + params.scaling_w * watermark_signal
        # 局部化：只在掩码区域应用水印
        imgs_w_local = combined_mask * imgs_w + (1 - combined_mask) * imgs

        # 3. 同步数据增强（确保图像与掩码一致变换，掩码仅做几何变换）
        imgs_aug, gt_mask_final, attack_name = water_attacker(imgs_w_local, combined_mask)

        # 4. 提取
        pred_mask, pred_msg_tensor = dec(imgs_aug)

        # 5. 计算损失 - 让消息解码依赖于掩码预测
        loss_det = detection_loss_iou(pred_mask, gt_mask_final)
        loss_dec = decoding_loss(pred_msg_tensor, gt_msg_tensor, pred_mask, gt_mask_final, use_pred_mask=True)
        
        loss = params.lambda_det * loss_det + params.lambda_dec * loss_dec

        # 6. 梯度更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 7. 统计信息
        # 计算解码准确率 - 在预测掩码区域内评估
        pred_msgs_prob = torch.sigmoid(pred_msg_tensor)  # logits -> [0,1]
        pred_msgs_bool = (pred_msgs_prob > 0.5).float()  # 二值化
        
        # 使用预测掩码加权的准确率：只在模型认为有水印的地方评估解码准确率
        pred_mask_binary = (pred_mask > 0.5).float()
        mask_intersection = pred_mask_binary * gt_mask_final  # 预测掩码与真实掩码的交集
        
        if mask_intersection.sum() > 0:
            correct_bits = ((pred_msgs_bool == gt_msg_tensor).float() * mask_intersection).sum()
            total_bits_in_intersection = (mask_intersection.sum() * params.num_bits)
            bit_acc = correct_bits / (total_bits_in_intersection + 1e-6)
        else:
            # 如果没有交集，回退到传统方法
            correct_bits = ((pred_msgs_bool == gt_msg_tensor).float() * gt_mask_final).sum()
            total_bits_in_mask = (gt_mask_final.sum() * params.num_bits)
            bit_acc = correct_bits / (total_bits_in_mask + 1e-6)

        # 计算真实IoU指标
        pred_mask_binary = (pred_mask > 0.5).float()
        intersection = (pred_mask_binary * gt_mask_final).sum()
        union = (pred_mask_binary + gt_mask_final).sum() - intersection
        iou_real = intersection / (union + 1e-6)
        
        log_stats = {
            'loss_det': loss_det.item(),
            'loss_dec': loss_dec.item(),
            'loss': loss.item(),
            'lr': optimizer.param_groups[0]['lr'],
            'bit_acc_avg': bit_acc.item(),
            'iou_avg': iou_real.item(),  # 真实IoU指标
        }
        
        metric_logger.update(**log_stats)

        # 8. 收集示例图像用于轮末保存
        if len(epoch_images['original']) < 8:  # 只收集前8张图像
            # 选择批次中的前几张图像
            num_to_collect = min(8 - len(epoch_images['original']), B)
            for i in range(num_to_collect):
                epoch_images['original'].append(imgs[i].cpu())
                epoch_images['gt_mask'].append(combined_mask[i].cpu())
                epoch_images['watermarked'].append(imgs_w_local[i].cpu())
                epoch_images['pred_mask'].append(torch.sigmoid(pred_mask[i]).cpu())

    # 9. 轮末保存8张图像，分别保存到4个页面
    if len(epoch_images['original']) > 0:
        output_dir = Path(params.output_dir) / 'images'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 转换掩码为RGB格式用于可视化
        gt_masks_rgb = [mask.repeat(3, 1, 1) if mask.dim() == 3 and mask.shape[0] == 1 else mask for mask in epoch_images['gt_mask']]
        pred_masks_rgb = [mask.repeat(3, 1, 1) if mask.dim() == 3 and mask.shape[0] == 1 else mask for mask in epoch_images['pred_mask']]
        
        # 保存4个页面，每页8张图像排列成2x4网格
        save_image(epoch_images['original'], output_dir / f'epoch_{epoch:03d}_01_original.png', 
                  nrow=4, normalize=True, pad_value=1)
        save_image(gt_masks_rgb, output_dir / f'epoch_{epoch:03d}_02_gt_masks.png', 
                  nrow=4, normalize=False, pad_value=1)
        save_image(epoch_images['watermarked'], output_dir / f'epoch_{epoch:03d}_03_watermarked.png', 
                  nrow=4, normalize=True, pad_value=1)
        save_image(pred_masks_rgb, output_dir / f'epoch_{epoch:03d}_04_pred_masks.png', 
                  nrow=4, normalize=False, pad_value=1)
        
        print(f"Saved epoch {epoch} images: {len(epoch_images['original'])} samples")

    metric_logger.synchronize_between_processes()
    print("Averaged {} stats:".format('train'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# main函数需要做重大修改
def main(params):
    # ... (保留分布式初始化和随机种子设置部分) ...
    if params.dist:
        utils.init_distributed_mode(params)
    seed = params.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print("__git__:{}".format(utils.get_sha()))
    print("__log__:{}".format(json.dumps(vars(params))))
    
    # ... (保留参数处理部分) ...

    # 1. 构建编码器 (Embedder_temp)
    print('building encoder (Embedder_temp)...')
    # 我们继续使用原有的Encoder作为我们的临时嵌入器
    encoder = models.HiddenEncoder(num_blocks=params.encoder_depth, num_bits=params.num_bits, channels=params.encoder_channels, last_tanh=params.use_tanh)

    # 2. 构建解码器 (W_seg)
    print('building decoder (W_seg)...')
    # CHANGED: 使用我们新的分割解码器
    decoder = models.SegmentationDecoder(num_bits=params.num_bits, channels=params.decoder_channels)
    
    # ... (保留BN momentum设置) ...
    
    # 3. 构建数据增强模块
    print('building data augmentation...')
    # 这个模块现在将同时用于图像和掩码
    data_aug = data_augmentation.HiddenAug(params.img_size, params.p_crop, params.p_blur,  params.p_jpeg, params.p_rot,  params.p_color_jitter, params.p_res).to(device)

    # 新增：用于图像-掩码同步的攻击增强器
    water_attacker = build_augmenter()

    # 4. 创建一个简化的 EncoderDecoder 来管理这两个模块
    # 注意：我们不再需要原始的EncoderDecoder类，但为了代码兼容性，我们用一个简单的容器
    # 实际的复杂逻辑移到了 train_one_epoch 中
    encoder_decoder = nn.ModuleDict({'encoder': encoder, 'decoder': decoder}).to(device)

    if params.dist:
        encoder_decoder = nn.SyncBatchNorm.convert_sync_batchnorm(encoder_decoder)
        encoder_decoder = nn.parallel.DistributedDataParallel(encoder_decoder, device_ids=[params.local_rank])
    
    # 5. 构建优化器和调度器
    # ... (保留优化器和调度器构建部分，它们基本不变) ...
    optim_params = utils.parse_params(params.optimizer)
    lr_mult = params.batch_size * utils.get_world_size() / 512.0
    optim_params['lr'] = lr_mult * optim_params['lr'] if 'lr' in optim_params else lr_mult * 1e-3
    to_optim = [*encoder.parameters(), *decoder.parameters()]
    optimizer = utils.build_optimizer(model_params=to_optim, **optim_params)
    scheduler = utils.build_lr_scheduler(optimizer=optimizer, **utils.parse_params(params.scheduler)) if params.scheduler is not None else None
    print('optimizer: %s'%optimizer)
    print('scheduler: %s'%scheduler)

    # 6. 数据加载器
    # ... (保留数据加载器构建部分，它们不变) ...
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(params.img_size, scale=(0.5, 1.0)), # 增加一些随机性
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
    if params.use_coco_segmentation and params.train_annotation_file is not None:
        train_loader = get_coco_segmentation_loader(
            data_dir=params.train_dir,
            ann_file=params.train_annotation_file,
            transform=train_transform,
            mask_transform=None,
            batch_size=params.batch_size,
            shuffle=True,
            num_workers=params.workers,
            random_nb_object=True,
            multi_w=False,
            max_nb_masks=params.max_nb_masks,
        )
        # 验证集可选，当前训练逻辑未用到
        if params.val_annotation_file is not None:
            val_loader = get_coco_segmentation_loader(
                data_dir=params.val_dir,
                ann_file=params.val_annotation_file,
                transform=val_transform,
                mask_transform=None,
                batch_size=params.batch_size_eval,
                shuffle=False,
                num_workers=params.workers,
                random_nb_object=False,
                multi_w=False,
                max_nb_masks=params.max_nb_masks,
            )
        else:
            val_loader = utils.get_dataloader(params.val_dir, transform=val_transform,  batch_size=params.batch_size_eval, num_workers=params.workers, shuffle=False)
    else:
        train_loader = utils.get_dataloader(params.train_dir, transform=train_transform, batch_size=params.batch_size, num_workers=params.workers, shuffle=True)
        val_loader = utils.get_dataloader(params.val_dir, transform=val_transform,  batch_size=params.batch_size_eval, num_workers=params.workers, shuffle=False)

    # 6.5 创建输出目录
    output_dir = Path(params.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存训练参数
    with open(output_dir / 'params.json', 'w') as f:
        json.dump(vars(params), f, indent=2)

    # 7. 开始训练
    start_epoch = 0
    print('training phase 1...')
    start_time = time.time()
    for epoch in range(start_epoch, params.epochs):
        if params.dist:
            train_loader.sampler.set_epoch(epoch)

        # 传入data_aug模块
        train_stats = train_one_epoch(encoder_decoder, train_loader, optimizer, scheduler, epoch, params, data_aug, water_attacker)
        
        # 评估部分暂时跳过，专注于训练
        if (epoch + 1) % params.eval_freq == 0:
            print(f"Epoch {epoch + 1}/{params.epochs} completed. Train stats: {train_stats}")
        
        # 保存检查点
        if (epoch + 1) % params.saveckp_freq == 0 or epoch == params.epochs - 1:
            checkpoint = {
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_stats': train_stats
            }
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch+1:03d}.pth')
            print(f"Checkpoint saved: checkpoint_epoch_{epoch+1:03d}.pth")

    total_time = time.time() - start_time
    print(f'Training completed in {total_time:.2f} seconds')
    
    # 保存最终模型
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'params': vars(params)
    }, output_dir / 'final_model.pth')
    print("Final model saved as 'final_model.pth'")

if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    main(params)