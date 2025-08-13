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

# NEW: 计算解码损失
def decoding_loss(pred_msg_tensor, gt_msg_tensor, gt_mask):
    # 只在掩码区域计算损失
    loss = F.binary_cross_entropy_with_logits(pred_msg_tensor, gt_msg_tensor, reduction='none')
    loss = (loss * gt_mask).sum() / (gt_mask.sum() * gt_msg_tensor.shape[1] + 1e-6)
    return loss

# CHANGED: 整个 train_one_epoch 函数被重写以适应新流程
def train_one_epoch(encoder_decoder: models.EncoderDecoder, loader, optimizer, scheduler, epoch, params, data_aug):
    if params.scheduler is not None:
        scheduler.step(epoch)
    encoder_decoder.train()
    header = 'Train - Epoch: [{}/{}]'.format(epoch, params.epochs)
    metric_logger = utils.MetricLogger(delimiter="  ")

    for it, (imgs, _) in enumerate(metric_logger.log_every(loader, 10, header)):
        imgs = imgs.to(device, non_blocking=True)
        B, C, H, W = imgs.shape

        # 1. 生成真值掩码和消息
        gt_mask = create_random_mask((H, W), B)
        msgs_ori = torch.rand((B, params.num_bits)) > 0.5
        msgs_float = msgs_ori.float().to(device)
        gt_msg_tensor = msgs_float.view(B, params.num_bits, 1, 1).expand(-1, -1, H, W)
        
        # 2. 局部化水印嵌入
        # encoder生成全局信号
        watermark_signal = encoder_decoder.module.encoder(imgs, 2 * msgs_float - 1)
        # 添加并局部化
        imgs_w = imgs + params.scaling_w * watermark_signal
        imgs_w_local = gt_mask * imgs_w + (1 - gt_mask) * imgs
        
        # 3. 同步数据增强
        # 堆叠图像和掩码以应用相同的随机变换
        stacked = torch.cat([imgs_w_local, gt_mask.expand(-1, 3, -1, -1)], dim=0)
        stacked_aug = data_aug(stacked)
        imgs_aug, gt_mask_final = torch.split(stacked_aug, B, dim=0)
        gt_mask_final = (gt_mask_final.mean(dim=1, keepdim=True) > 0.5).float() # 从3通道转回1通道

        # 4. 提取
        pred_mask, pred_msg_tensor = encoder_decoder.module.decoder(imgs_aug)

        # 5. 计算损失
        loss_det = detection_loss_iou(pred_mask, gt_mask_final)
        loss_dec = decoding_loss(pred_msg_tensor, gt_msg_tensor, gt_mask_final)
        
        loss = params.lambda_det * loss_det + params.lambda_dec * loss_dec

        # 6. 梯度更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 7. 统计信息
        # 计算解码准确率
        pred_msgs_bool = (pred_msg_tensor > 0).float()
        correct_bits = ((pred_msgs_bool == gt_msg_tensor).float() * gt_mask_final).sum()
        total_bits_in_mask = (gt_mask_final.sum() * params.num_bits)
        bit_acc = correct_bits / (total_bits_in_mask + 1e-6)

        log_stats = {
            'loss_det': loss_det.item(),
            'loss_dec': loss_dec.item(),
            'loss': loss.item(),
            'lr': optimizer.param_groups[0]['lr'],
            'bit_acc_avg': bit_acc.item(),
            'iou_avg': 1 - loss_det.item(), # 从损失反推IoU
        }
        
        metric_logger.update(**log_stats)

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
    train_loader = utils.get_dataloader(params.train_dir, transform=train_transform, batch_size=params.batch_size, num_workers=params.workers, shuffle=True)
    val_loader = utils.get_dataloader(params.val_dir, transform=val_transform,  batch_size=params.batch_size_eval, num_workers=params.workers, shuffle=False)

    # ... (保留恢复训练和创建输出目录的部分) ...

    # 7. 开始训练
    print('training phase 1...')
    start_time = time.time()
    for epoch in range(start_epoch, params.epochs):
        if params.dist:
            train_loader.sampler.set_epoch(epoch)

        # 传入data_aug模块
        train_stats = train_one_epoch(encoder_decoder, train_loader, optimizer, scheduler, epoch, params, data_aug)
        
        # (评估部分eval_one_epoch也需要类似地重写，这里为简化省略，但逻辑与训练类似)
        
        # ... (保留保存模型和日志的部分) ...

if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    main(params)
