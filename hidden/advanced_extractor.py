# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import DBSCAN
from typing import Tuple, Dict, Optional

# U-Net based extractor components
class UNetBlock(nn.Module):
    """Basic U-Net block with conv-bn-relu"""
    def __init__(self, in_channels, out_channels, downsample=False):
        super(UNetBlock, self).__init__()
        self.downsample = downsample
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if downsample:
            self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        if self.downsample:
            return self.pool(x), x  # return pooled and unpooled for skip connection
        return x

class UNetUpBlock(nn.Module):
    """U-Net upsampling block with skip connections"""
    def __init__(self, in_channels, out_channels):
        super(UNetUpBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv_block = UNetBlock(out_channels * 2, out_channels)
        
    def forward(self, x, skip):
        x = self.upsample(x)
        # Match spatial dimensions if needed
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x

class ViTEncoder(nn.Module):
    """Vision Transformer encoder adapted from watermark-anything"""
    def __init__(self, img_size=128, patch_size=16, in_chans=3, embed_dim=384, depth=6, num_heads=6):
        super(ViTEncoder, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embedding
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(depth)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # B, embed_dim, H//patch_size, W//patch_size
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Reshape back to spatial format
        h, w = H // self.patch_size, W // self.patch_size
        x = x.transpose(1, 2).reshape(B, self.embed_dim, h, w)
        
        return x

class PixelDecoder(nn.Module):
    """Pixel decoder for generating pixel-level predictions"""
    def __init__(self, embed_dim, num_bits, upscale_factor=16):
        super(PixelDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_bits = num_bits
        self.upscale_factor = upscale_factor
        
        # Progressive upsampling layers
        current_dim = embed_dim
        self.upsampling_layers = nn.ModuleList()
        
        # Calculate number of upsampling stages
        stages = int(np.log2(upscale_factor))
        for i in range(stages):
            next_dim = current_dim // 2
            self.upsampling_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(current_dim, next_dim, 4, stride=2, padding=1),
                    nn.BatchNorm2d(next_dim),
                    nn.ReLU(inplace=True)
                )
            )
            current_dim = next_dim
        
        # Final prediction layer: 1 channel for mask + num_bits for messages
        self.final_conv = nn.Conv2d(current_dim, 1 + num_bits, 1)
        
    def forward(self, x):
        # Progressive upsampling
        for layer in self.upsampling_layers:
            x = layer(x)
        
        # Final prediction
        output = self.final_conv(x)
        return output

class MultiWatermarkDBSCAN:
    """DBSCAN-based multi-watermark clustering adapted from watermark-anything"""
    def __init__(self, epsilon=1.0, min_samples=100):
        self.epsilon = epsilon
        self.min_samples = min_samples
    
    def cluster_watermarks(self, bit_preds, mask_preds, threshold=0.0):
        """
        Perform DBSCAN clustering on predicted bits to identify multiple watermarks
        
        Args:
            bit_preds: torch.Tensor [B, num_bits, H, W] - predicted bits per pixel
            mask_preds: torch.Tensor [B, 1, H, W] - predicted mask
            threshold: float - threshold for bit predictions
            
        Returns:
            tuple: (centroids dict, cluster_labels tensor)
        """
        bit_preds = bit_preds > threshold  # Binarize predictions
        B, num_bits, H, W = bit_preds.shape
        
        all_centroids = []
        all_labels = []
        
        for b in range(B):
            mask = (mask_preds[b, 0] > 0.5).float()  # H, W
            bits = bit_preds[b]  # num_bits, H, W
            
            # Flatten and get valid pixels
            bits_flat = bits.view(num_bits, -1).t().cpu().numpy()  # H*W, num_bits
            valid_indices = (mask.view(-1) > 0).cpu().numpy()
            
            if valid_indices.sum() == 0:
                # No valid pixels
                all_centroids.append({})
                all_labels.append(torch.full((H, W), -1, dtype=torch.float32))
                continue
            
            valid_bits = bits_flat[valid_indices]  # num_valid, num_bits
            
            # Apply DBSCAN
            if valid_bits.shape[0] < self.min_samples:
                # Too few samples for clustering
                all_centroids.append({})
                all_labels.append(torch.full((H, W), -1, dtype=torch.float32))
                continue
                
            dbscan = DBSCAN(eps=self.epsilon, min_samples=self.min_samples)
            cluster_labels = dbscan.fit_predict(valid_bits)
            
            # Compute centroids for each cluster
            unique_labels = np.unique(cluster_labels)
            unique_labels = unique_labels[unique_labels != -1]  # Exclude noise
            
            centroids = {}
            for label in unique_labels:
                cluster_points = valid_bits[cluster_labels == label]
                centroid = cluster_points.mean(axis=0) > 0.5  # Binarize centroid
                centroids[int(label)] = torch.tensor(centroid, dtype=torch.bool)
            
            # Map labels back to spatial format
            full_labels = np.full(H * W, -1, dtype=np.float32)
            full_labels[valid_indices] = cluster_labels
            full_labels = torch.tensor(full_labels.reshape(H, W))
            
            all_centroids.append(centroids)
            all_labels.append(full_labels)
        
        return all_centroids, torch.stack(all_labels)

class AdvancedWatermarkExtractor(nn.Module):
    """
    Advanced watermark extractor with multiple architecture options and DBSCAN clustering
    """
    def __init__(self, num_bits, channels=64, img_size=128, architecture='unet', 
                 dbscan_epsilon=1.0, dbscan_min_samples=100):
        super(AdvancedWatermarkExtractor, self).__init__()
        self.num_bits = num_bits
        self.architecture = architecture
        self.img_size = img_size
        
        if architecture == 'unet':
            self._build_unet(channels)
        elif architecture == 'vit':
            self._build_vit()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # DBSCAN clustering for multi-watermark detection
        self.dbscan_clusterer = MultiWatermarkDBSCAN(
            epsilon=dbscan_epsilon, 
            min_samples=dbscan_min_samples
        )
        
    def _build_unet(self, channels):
        """Build U-Net architecture"""
        # Encoder path
        self.enc1 = UNetBlock(3, channels)
        self.enc2 = UNetBlock(channels, channels * 2, downsample=True)
        self.enc3 = UNetBlock(channels * 2, channels * 4, downsample=True)
        self.enc4 = UNetBlock(channels * 4, channels * 8, downsample=True)
        
        # Bottleneck
        self.bottleneck = UNetBlock(channels * 8, channels * 16, downsample=True)
        
        # Decoder path
        self.dec4 = UNetUpBlock(channels * 16, channels * 8)
        self.dec3 = UNetUpBlock(channels * 8, channels * 4)
        self.dec2 = UNetUpBlock(channels * 4, channels * 2)
        self.dec1 = UNetUpBlock(channels * 2, channels)
        
        # Final prediction layer
        self.final_conv = nn.Conv2d(channels, 1 + self.num_bits, 1)
        
    def _build_vit(self):
        """Build ViT + PixelDecoder architecture"""
        self.vit_encoder = ViTEncoder(
            img_size=self.img_size, 
            patch_size=16, 
            embed_dim=384, 
            depth=6
        )
        self.pixel_decoder = PixelDecoder(
            embed_dim=384, 
            num_bits=self.num_bits, 
            upscale_factor=16
        )
        
    def forward_unet(self, x):
        """Forward pass for U-Net architecture"""
        # Encoder path with skip connections
        x1 = self.enc1(x)
        x2, skip1 = self.enc2(x1)
        x3, skip2 = self.enc3(x2)
        x4, skip3 = self.enc4(x3)
        
        # Bottleneck
        x5, _ = self.bottleneck(x4)
        
        # Decoder path
        x = self.dec4(x5, skip3)
        x = self.dec3(x, skip2)
        x = self.dec2(x, skip1)
        x = self.dec1(x, x1)
        
        # Final prediction
        output = self.final_conv(x)
        return output
        
    def forward_vit(self, x):
        """Forward pass for ViT architecture"""
        # ViT encoding
        features = self.vit_encoder(x)
        
        # Pixel decoding
        output = self.pixel_decoder(features)
        return output
        
    def forward(self, x):
        """
        Forward pass of the extractor
        
        Args:
            x: torch.Tensor [B, 3, H, W] - input images
            
        Returns:
            dict: {
                'mask_pred': torch.Tensor [B, 1, H, W] - predicted watermark mask
                'bit_pred': torch.Tensor [B, num_bits, H, W] - predicted bits
                'centroids': list of dicts - DBSCAN cluster centroids
                'cluster_labels': torch.Tensor [B, H, W] - cluster assignment labels
            }
        """
        if self.architecture == 'unet':
            output = self.forward_unet(x)
        else:
            output = self.forward_vit(x)
        
        # Split output into mask and bit predictions
        mask_pred = output[:, 0:1, :, :]  # B, 1, H, W
        bit_pred = output[:, 1:, :, :] if self.num_bits > 0 else None  # B, num_bits, H, W
        
        result = {
            'mask_pred': mask_pred,
            'bit_pred': bit_pred
        }
        
        # Apply DBSCAN clustering if we have bit predictions
        if bit_pred is not None:
            centroids, cluster_labels = self.dbscan_clusterer.cluster_watermarks(
                bit_pred, mask_pred
            )
            result['centroids'] = centroids
            result['cluster_labels'] = cluster_labels
        
        return result
    
    def extract_messages(self, x, use_clustering=True):
        """
        Extract watermark messages from input images
        
        Args:
            x: torch.Tensor [B, 3, H, W] - input images
            use_clustering: bool - whether to use DBSCAN clustering
            
        Returns:
            dict: extracted watermark information
        """
        with torch.no_grad():
            outputs = self.forward(x)
            
            if not use_clustering or self.num_bits == 0:
                # Simple global averaging like original stable_signature
                mask_pred = torch.sigmoid(outputs['mask_pred'])
                if outputs['bit_pred'] is not None:
                    bit_pred = outputs['bit_pred']
                    # Global average pooling weighted by mask
                    weighted_bits = bit_pred * mask_pred
                    global_bits = F.adaptive_avg_pool2d(weighted_bits, (1, 1)).squeeze(-1).squeeze(-1)
                    outputs['global_message'] = torch.sign(global_bits)
                
                return outputs
            else:
                # Use DBSCAN clustering results
                return outputs

def build_advanced_extractor(num_bits, architecture='unet', **kwargs):
    """
    Factory function to build advanced watermark extractor
    
    Args:
        num_bits: int - number of watermark bits
        architecture: str - 'unet' or 'vit'
        **kwargs: additional arguments for the extractor
        
    Returns:
        AdvancedWatermarkExtractor instance
    """
    return AdvancedWatermarkExtractor(
        num_bits=num_bits,
        architecture=architecture,
        **kwargs
    )

