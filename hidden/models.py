# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F
import torch.nn as nn
from timm.models import vision_transformer

import attenuations
from advanced_extractor import AdvancedWatermarkExtractor, build_advanced_extractor
from augmentation_attacks import WatermarkAugmenter, build_augmenter

class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out):

        super(ConvBNRelu, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels_out, eps=1e-3),
            nn.GELU()
        )

    def forward(self, x):
        return self.layers(x)

class HiddenEncoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, num_blocks, num_bits, channels, last_tanh=True):
        super(HiddenEncoder, self).__init__()
        layers = [ConvBNRelu(3, channels)]

        for _ in range(num_blocks-1):
            layer = ConvBNRelu(channels, channels)
            layers.append(layer)

        self.conv_bns = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(channels + 3 + num_bits, channels)

        self.final_layer = nn.Conv2d(channels, 3, kernel_size=1)

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, imgs, msgs):

        msgs = msgs.unsqueeze(-1).unsqueeze(-1) # b l 1 1
        msgs = msgs.expand(-1,-1, imgs.size(-2), imgs.size(-1)) # b l h w

        encoded_image = self.conv_bns(imgs) # b c h w

        concat = torch.cat([msgs, encoded_image, imgs], dim=1) # b l+c+3 h w
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)

        if self.last_tanh:
            im_w = self.tanh(im_w)

        return im_w

class HiddenDecoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, num_blocks, num_bits, channels):

        super(HiddenDecoder, self).__init__()

        layers = [ConvBNRelu(3, channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBNRelu(channels, channels))

        layers.append(ConvBNRelu(channels, num_bits))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(num_bits, num_bits)

    def forward(self, img_w):

        x = self.layers(img_w) # b d 1 1
        x = x.squeeze(-1).squeeze(-1) # b d
        x = self.linear(x) # b d
        return x

class AdvancedHiddenDecoder(nn.Module):
    """
    Advanced watermark decoder that supports both traditional bit extraction and 
    multi-watermark segmentation with DBSCAN clustering.
    """
    def __init__(self, num_bits, channels=64, img_size=128, architecture='unet', 
                 use_advanced=True, dbscan_epsilon=1.0, dbscan_min_samples=100):
        super(AdvancedHiddenDecoder, self).__init__()
        self.num_bits = num_bits
        self.use_advanced = use_advanced
        
        if use_advanced:
            # Use the new advanced extractor
            self.extractor = AdvancedWatermarkExtractor(
                num_bits=num_bits,
                channels=channels,
                img_size=img_size,
                architecture=architecture,
                dbscan_epsilon=dbscan_epsilon,
                dbscan_min_samples=dbscan_min_samples
            )
        else:
            # Fall back to original HiddenDecoder
            self.original_decoder = HiddenDecoder(
                num_blocks=8, 
                num_bits=num_bits, 
                channels=channels
            )
    
    def forward(self, img_w, return_detailed=False):
        """
        Forward pass that can return either traditional bit vector or detailed segmentation
        
        Args:
            img_w: torch.Tensor [B, 3, H, W] - watermarked images
            return_detailed: bool - whether to return detailed segmentation results
            
        Returns:
            If return_detailed=False: torch.Tensor [B, num_bits] - extracted bits (compatible with original)
            If return_detailed=True: dict with detailed extraction results
        """
        if not self.use_advanced:
            # Use original decoder
            return self.original_decoder(img_w)
        
        # Use advanced extractor
        outputs = self.extractor(img_w)
        
        if not return_detailed:
            # Return format compatible with original decoder
            if outputs['bit_pred'] is not None:
                # Global average pooling weighted by mask for compatibility
                mask_pred = torch.sigmoid(outputs['mask_pred'])
                bit_pred = outputs['bit_pred']
                weighted_bits = bit_pred * mask_pred
                global_bits = torch.adaptive_avg_pool2d(weighted_bits, (1, 1)).squeeze(-1).squeeze(-1)
                return global_bits
            else:
                # No bits to extract
                return torch.zeros(img_w.shape[0], self.num_bits, device=img_w.device)
        else:
            # Return detailed results including segmentation and clustering
            return outputs
    
    def extract_multiple_watermarks(self, img_w):
        """
        Extract multiple watermarks using DBSCAN clustering
        
        Args:
            img_w: torch.Tensor [B, 3, H, W] - watermarked images
            
        Returns:
            dict: detailed extraction results with clustering information
        """
        if not self.use_advanced:
            raise RuntimeError("Multi-watermark extraction requires use_advanced=True")
        
        return self.extractor.extract_messages(img_w, use_clustering=True)

class ImgEmbed(nn.Module):
    """ Patch to Image Embedding
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.ConvTranspose2d(embed_dim, in_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, num_patches_w, num_patches_h):
        B, S, CKK = x.shape # ckk = embed_dim
        x = self.proj(x.transpose(1,2).reshape(B, CKK, num_patches_h, num_patches_w)) # b s (c k k) -> b (c k k) s -> b (c k k) sh sw -> b c h w
        return x

class VitEncoder(vision_transformer.VisionTransformer):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, num_bits, last_tanh=True, **kwargs):
        super(VitEncoder, self).__init__(**kwargs)

        self.head = nn.Identity()
        self.norm = nn.Identity()

        self.msg_linear = nn.Linear(self.embed_dim+num_bits, self.embed_dim)

        self.unpatch = ImgEmbed(embed_dim=self.embed_dim, patch_size=kwargs['patch_size'])

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, x, msgs):

        num_patches = int(self.patch_embed.num_patches**0.5)

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        msgs = msgs.unsqueeze(1) # b 1 k
        msgs = msgs.repeat(1, x.shape[1], 1) # b 1 k -> b l k
        for ii, blk in enumerate(self.blocks):
            x = torch.concat([x, msgs], dim=-1) # b l (cpq+k)
            x = self.msg_linear(x)
            x = blk(x)

        x = x[:, 1:, :] # without cls token
        img_w = self.unpatch(x, num_patches, num_patches)

        if self.last_tanh:
            img_w = self.tanh(img_w)

        return img_w

class DvmarkEncoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, num_blocks, num_bits, channels, last_tanh=True):
        super(DvmarkEncoder, self).__init__()

        transform_layers = [ConvBNRelu(3, channels)]
        for _ in range(num_blocks-1):
            layer = ConvBNRelu(channels, channels)
            transform_layers.append(layer)
        self.transform_layers = nn.Sequential(*transform_layers)

        # conv layers for original scale
        num_blocks_scale1 = 3
        scale1_layers = [ConvBNRelu(channels+num_bits, channels*2)]
        for _ in range(num_blocks_scale1-1):
            layer = ConvBNRelu(channels*2, channels*2)
            scale1_layers.append(layer)
        self.scale1_layers = nn.Sequential(*scale1_layers)

        # downsample x2
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # conv layers for downsampled
        num_blocks_scale2 = 3
        scale2_layers = [ConvBNRelu(channels*2+num_bits, channels*4), ConvBNRelu(channels*4, channels*2)]
        for _ in range(num_blocks_scale2-2):
            layer = ConvBNRelu(channels*2, channels*2)
            scale2_layers.append(layer)
        self.scale2_layers = nn.Sequential(*scale2_layers)

        # upsample x2
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.final_layer = nn.Conv2d(channels*2, 3, kernel_size=1)

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, imgs, msgs):

        encoded_image = self.transform_layers(imgs) # b c h w

        msgs = msgs.unsqueeze(-1).unsqueeze(-1) # b l 1 1

        scale1 = torch.cat([msgs.expand(-1,-1, imgs.size(-2), imgs.size(-1)), encoded_image], dim=1) # b l+c h w
        scale1 = self.scale1_layers(scale1) # b c*2 h w

        scale2 = self.avg_pool(scale1) # b c*2 h/2 w/2
        scale2 = torch.cat([msgs.expand(-1,-1, imgs.size(-2)//2, imgs.size(-1)//2), scale2], dim=1) # b l+c*2 h/2 w/2
        scale2 = self.scale2_layers(scale2) # b c*2 h/2 w/2

        scale1 = scale1 + self.upsample(scale2) # b c*2 h w
        im_w = self.final_layer(scale1) # b 3 h w

        if self.last_tanh:
            im_w = self.tanh(im_w)

        return im_w

class EncoderDecoder(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module, 
        attenuation: attenuations.JND, 
        augmentation: nn.Module, 
        decoder:nn.Module,
        scale_channels: bool,
        scaling_i: float,
        scaling_w: float,
        num_bits: int,
        redundancy: int
    ):
        super().__init__()
        self.encoder = encoder
        self.attenuation = attenuation
        self.augmentation = augmentation
        self.decoder = decoder
        # params for the forward pass
        self.scale_channels = scale_channels
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w
        self.num_bits = num_bits
        self.redundancy = redundancy

    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor,
        eval_mode: bool=False,
        eval_aug: nn.Module=nn.Identity(),
    ):
        """
        Does the full forward pass of the encoder-decoder network:
        - encodes the message into the image
        - attenuates the watermark
        - augments the image
        - decodes the watermark

        Args:
            imgs: b c h w
            msgs: b l
        """

        # encoder
        deltas_w = self.encoder(imgs, msgs) # b c h w

        # scaling channels: more weight to blue channel
        if self.scale_channels:
            aa = 1/4.6 # such that aas has mean 1
            aas = torch.tensor([aa*(1/0.299), aa*(1/0.587), aa*(1/0.114)]).to(imgs.device) 
            deltas_w = deltas_w * aas[None,:,None,None]

        # add heatmaps
        if self.attenuation is not None:
            heatmaps = self.attenuation.heatmaps(imgs) # b 1 h w
            deltas_w = deltas_w * heatmaps # # b c h w * b 1 h w -> b c h w
        imgs_w = self.scaling_i * imgs + self.scaling_w * deltas_w # b c h w

        # data augmentation
        if eval_mode:
            imgs_aug = eval_aug(imgs_w)
            fts = self.decoder(imgs_aug) # b c h w -> b d
        else:
            imgs_aug = self.augmentation(imgs_w)
            fts = self.decoder(imgs_aug) # b c h w -> b d
            
        fts = fts.view(-1, self.num_bits, self.redundancy) # b k*r -> b k r
        fts = torch.sum(fts, dim=-1) # b k r -> b k

        return fts, (imgs_w, imgs_aug)

class EncoderWithJND(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module, 
        attenuation: attenuations.JND, 
        scale_channels: bool,
        scaling_i: float,
        scaling_w: float
    ):
        super().__init__()
        self.encoder = encoder
        self.attenuation = attenuation
        # params for the forward pass
        self.scale_channels = scale_channels
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w

    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor,
    ):
        """ Does the forward pass of the encoder only """

        # encoder
        deltas_w = self.encoder(imgs, msgs) # b c h w

        # scaling channels: more weight to blue channel
        if self.scale_channels:
            aa = 1/4.6 # such that aas has mean 1
            aas = torch.tensor([aa*(1/0.299), aa*(1/0.587), aa*(1/0.114)]).to(imgs.device) 
            deltas_w = deltas_w * aas[None,:,None,None]

        # add heatmaps
        if self.attenuation is not None:
            heatmaps = self.attenuation.heatmaps(imgs) # b 1 h w
            deltas_w = deltas_w * heatmaps # # b c h w * b 1 h w -> b c h w
        imgs_w = self.scaling_i * imgs + self.scaling_w * deltas_w # b c h w

        return imgs_w


class LocalizedEncoderDecoder(nn.Module):
    """
    支持局部化水印、多水印嵌入和攻击模拟的编码器-解码器
    参考watermark-anything项目实现
    """
    def __init__(
        self, 
        encoder: nn.Module, 
        attenuation: attenuations.JND, 
        augmentation: nn.Module, 
        decoder: nn.Module,
        scale_channels: bool,
        scaling_i: float,
        scaling_w: float,
        num_bits: int,
        redundancy: int,
        max_watermarks: int = 4,
        use_localized: bool = True,
        use_advanced_attacks: bool = True,
        attack_config: dict = None
    ):
        super().__init__()
        self.encoder = encoder
        self.attenuation = attenuation
        self.augmentation = augmentation  # 原始增强
        self.decoder = decoder
        
        # params for the forward pass
        self.scale_channels = scale_channels
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w
        self.num_bits = num_bits
        self.redundancy = redundancy
        self.max_watermarks = max_watermarks
        self.use_localized = use_localized
        
        # 新增：攻击模拟器
        if use_advanced_attacks:
            self.attack_augmenter = build_augmenter(attack_config)
        else:
            self.attack_augmenter = None

    def get_random_msg(self, batch_size: int, device: torch.device):
        """生成随机消息"""
        msgs_ori = torch.rand((batch_size, self.num_bits), device=device) > 0.5
        msgs = 2 * msgs_ori.type(torch.float) - 1  # 转换为 -1/1
        return msgs

    def embed_watermark(self, imgs: torch.Tensor, msgs: torch.Tensor):
        """嵌入水印到图像中"""
        # encoder
        deltas_w = self.encoder(imgs, msgs)  # b c h w

        # scaling channels: more weight to blue channel
        if self.scale_channels:
            aa = 1/4.6  # such that aas has mean 1
            aas = torch.tensor([aa*(1/0.299), aa*(1/0.587), aa*(1/0.114)]).to(imgs.device) 
            deltas_w = deltas_w * aas[None, :, None, None]

        # add heatmaps (JND attenuation)
        if self.attenuation is not None:
            heatmaps = self.attenuation.heatmaps(imgs)  # b 1 h w
            deltas_w = deltas_w * heatmaps  # b c h w * b 1 h w -> b c h w
        
        # 生成全尺寸水印图像
        imgs_w_full = self.scaling_i * imgs + self.scaling_w * deltas_w  # b c h w
        
        return imgs_w_full, deltas_w

    def localize_watermark(self, imgs_clean: torch.Tensor, imgs_w_full: torch.Tensor, 
                          gt_masks: torch.Tensor):
        """
        局部化水印到指定区域
        
        Args:
            imgs_clean: [B, C, H, W] - 干净图像
            imgs_w_full: [B, C, H, W] - 全尺寸水印图像（单个水印）
            gt_masks: [B, 1, H, W] 或 [B, K, H, W] - 真值掩码
            
        Returns:
            imgs_w_local: [B, C, H, W] - 局部化水印图像
        """
        if not self.use_localized:
            return imgs_w_full
        
        if len(gt_masks.shape) == 4 and gt_masks.shape[1] == 1:
            # 单一掩码情况 [B, 1, H, W]
            imgs_w_local = gt_masks * imgs_w_full + (1 - gt_masks) * imgs_clean
        else:
            # 单个水印应用到第一个掩码 
            mask_first = gt_masks[:, 0:1, :, :]  # [B, 1, H, W]
            imgs_w_local = mask_first * imgs_w_full + (1 - mask_first) * imgs_clean
        
        return imgs_w_local
        
    def multi_watermark_embed(self, imgs_clean: torch.Tensor, gt_masks: torch.Tensor):
        """
        为多个掩码区域生成和嵌入独立的32bit水印
        参考watermark-anything项目的多水印嵌入实现
        
        Args:
            imgs_clean: [B, C, H, W] - 干净图像
            gt_masks: [B, K, H, W] - K个掩码区域
            
        Returns:
            dict: {
                'imgs_multi_w': [B, C, H, W] - 多水印图像,
                'msgs_list': List[Tensor] - 每个水印的消息,
                'combined_mask': [B, 1, H, W] - 组合掩码
            }
        """
        batch_size = imgs_clean.shape[0]
        num_masks = gt_masks.shape[1]
        
        # 多水印嵌入：为每个掩码区域生成独立的32bit水印
        
        msgs_list = []
        combined_imgs = imgs_clean.clone()
        combined_mask = torch.zeros_like(gt_masks[:, 0:1, :, :])  # [B, 1, H, W]
        
        # 为每个掩码区域生成和嵌入独立的32bit水印
        for mask_idx in range(num_masks):
            # 生成独立的32bit消息
            msgs_k = self.get_random_msg(batch_size, imgs_clean.device)
            msgs_list.append(msgs_k)
            
            # 获取当前掩码
            current_mask = gt_masks[:, mask_idx:mask_idx+1, :, :]  # [B, 1, H, W]
            
            # 为当前掩码生成水印
            imgs_w_k, deltas_w_k = self.embed_watermark(imgs_clean, msgs_k)
            
            # 局部化当前水印到掩码区域
            # 只在掩码区域应用水印，其他区域保持当前状态
            combined_imgs = combined_imgs * (1 - current_mask) + imgs_w_k * current_mask
            
            # 更新组合掩码
            combined_mask = torch.clamp(combined_mask + current_mask, 0, 1)
        
        return {
            'imgs_multi_w': combined_imgs,
            'msgs_list': msgs_list,
            'combined_mask': combined_mask
        }

    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor = None,
        gt_masks: torch.Tensor = None,
        eval_mode: bool = False,
        eval_aug: nn.Module = nn.Identity(),
        return_detailed: bool = False,
    ):
        """
        完整的前向传播，实现您描述的训练流程：
        a. 生成水印信号
        b. 添加水印  
        c. 局部化水印
        d. 模拟攻击
        e. 提取水印
        
        Args:
            imgs: [B, C, H, W] - 干净图像 (x_clean)
            msgs: [B, num_bits] - 真值消息 (gt_msg_tensor)，可选
            gt_masks: [B, 1, H, W] 或 [B, K, H, W] - 真值掩码，可选
            eval_mode: bool - 是否为评估模式
            eval_aug: nn.Module - 评估时的增强
            return_detailed: bool - 是否返回详细结果
            
        Returns:
            根据decoder类型返回不同格式：
            - 传统decoder: (fts, (imgs_w_local, imgs_aug))
            - 高级decoder: 详细的分割和聚类结果
        """
        batch_size = imgs.shape[0]
        
        # 检查是否为多掩码情况，支持多个32bit水印嵌入
        if gt_masks is not None and len(gt_masks.shape) == 4 and gt_masks.shape[1] > 1:
            # 多掩码情况：为每个掩码区域生成独立的32bit水印
            multi_wm_results = self.multi_watermark_embed(imgs, gt_masks)
            imgs_w_local = multi_wm_results['imgs_multi_w']
            msgs_list = multi_wm_results['msgs_list']
            combined_mask = multi_wm_results['combined_mask']
            
            # 使用第一个消息作为主要消息（用于兼容性）
            msgs_primary = msgs_list[0] if msgs_list else None
            gt_masks_for_attack = combined_mask
            
        else:
            # 单掩码或无掩码情况：使用原有逻辑
            if msgs is None:
                msgs_primary = self.get_random_msg(batch_size, imgs.device)
            else:
                msgs_primary = msgs
            
            # a. 生成水印信号 & b. 添加水印
            imgs_w_full, deltas_w = self.embed_watermark(imgs, msgs_primary)
            
            # c. 局部化水印（如果提供了掩码）
            if gt_masks is not None and self.use_localized:
                imgs_w_local = self.localize_watermark(imgs, imgs_w_full, gt_masks)
                gt_masks_for_attack = gt_masks[:, 0:1, :, :] if len(gt_masks.shape) == 4 else gt_masks
            else:
                imgs_w_local = imgs_w_full
                gt_masks_for_attack = gt_masks
            
            msgs_list = [msgs_primary]
        
        # d. 模拟攻击
        if eval_mode:
            # 评估模式：使用指定的增强
            imgs_aug = eval_aug(imgs_w_local)
            gt_mask_final = gt_masks_for_attack
        else:
            # 训练模式：使用攻击增强器
            if self.attack_augmenter is not None and gt_masks_for_attack is not None:
                # 使用高级攻击增强器
                imgs_aug, gt_mask_final, attack_name = self.attack_augmenter(imgs_w_local, gt_masks_for_attack)
            else:
                # 使用原始增强器
                imgs_aug = self.augmentation(imgs_w_local)
                gt_mask_final = gt_masks_for_attack
        
        # e. 提取水印
        if isinstance(self.decoder, AdvancedHiddenDecoder):
            # 使用高级解码器，支持分割输出
            if return_detailed:
                decoder_outputs = self.decoder(imgs_aug, return_detailed=True)

                # 为训练的统一性补充向量形式fts（用于bit_acc等指标）
                if decoder_outputs is not None and 'bit_pred' in decoder_outputs and decoder_outputs['bit_pred'] is not None:
                    mask_pred_logits = decoder_outputs.get('mask_pred', None)
                    bit_pred = decoder_outputs['bit_pred']  # [B, K, H, W]
                    if mask_pred_logits is not None:
                        mask_prob = torch.sigmoid(mask_pred_logits)
                        weighted_bits = bit_pred * mask_prob
                    else:
                        weighted_bits = bit_pred
                    fts_vec = F.adaptive_avg_pool2d(weighted_bits, (1, 1)).squeeze(-1).squeeze(-1)  # [B, K]
                else:
                    fts_vec = torch.zeros(batch_size, self.num_bits, device=imgs.device)

                # 添加额外信息到输出
                decoder_outputs.update({
                    'fts': fts_vec,
                    'gt_masks': gt_masks,
                    'gt_mask_final': gt_mask_final,
                    'msgs_gt': msgs_primary,
                    'msgs_list': msgs_list,  # 所有水印消息列表
                    'imgs_w_local': imgs_w_local,
                    'imgs_aug': imgs_aug,
                    'num_watermarks': len(msgs_list),
                })

                return decoder_outputs
            else:
                # 兼容模式：返回传统格式
                fts = self.decoder(imgs_aug, return_detailed=False)
        else:
            # 使用传统解码器
            fts = self.decoder(imgs_aug)
            
        # 处理冗余编码
        if hasattr(self, 'redundancy') and self.redundancy > 1:
            fts = fts.view(-1, self.num_bits, self.redundancy)  # b k*r -> b k r
            fts = torch.sum(fts, dim=-1)  # b k r -> b k

        # 准备输出
        output_dict = {
            'fts': fts,
            'imgs_w_local': imgs_w_local,
            'imgs_aug': imgs_aug,
            'msgs_gt': msgs,
            'gt_masks': gt_masks,
            'gt_mask_final': gt_mask_final,
        }
        
        if return_detailed:
            return output_dict
        else:
            # 保持与原始EncoderDecoder兼容的输出格式
            return fts, (imgs_w_local, imgs_aug)

    def embed_multi_watermarks(
        self,
        imgs: torch.Tensor,
        gt_masks: torch.Tensor,
        msgs_list: list = None
    ):
        """
        多水印嵌入功能
        
        Args:
            imgs: [B, C, H, W] - 干净图像
            gt_masks: [B, K, H, W] - 多个掩码
            msgs_list: list of [B, num_bits] - 多个消息，可选
            
        Returns:
            dict: 包含多水印嵌入结果
        """
        batch_size, num_masks = gt_masks.shape[0], gt_masks.shape[1]
        
        if msgs_list is None:
            # 为每个水印生成独立消息
            msgs_list = [self.get_random_msg(batch_size, imgs.device) for _ in range(num_masks)]
        
        # 初始化组合图像
        combined_imgs = imgs.clone()
        combined_mask = torch.zeros_like(gt_masks[:, 0:1, :, :])
        
        for k in range(num_masks):
            mask_k = gt_masks[:, k:k+1, :, :]  # [B, 1, H, W]
            msgs_k = msgs_list[k]
            
            # 为当前掩码生成水印
            imgs_w_k, _ = self.embed_watermark(imgs, msgs_k)
            
            # 局部化到掩码区域
            combined_imgs = mask_k * imgs_w_k + (1 - mask_k) * combined_imgs
            combined_mask = torch.maximum(combined_mask, mask_k)
        
        return {
            'imgs_multi_w': combined_imgs,
            'combined_mask': combined_mask,
            'msgs_list': msgs_list,
            'individual_masks': gt_masks
        }
        
