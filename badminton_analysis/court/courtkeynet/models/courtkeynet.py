"""
CourtKeyNet Main Architecture
Manuscript: Section 3, Figure 2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .octave import OctaveFeatureExtractor
from .polar import PolarTransformAttention
from .qcm import QuadrilateralConstraintModule


class CourtKeyNet(nn.Module):
    """
    Complete CourtKeyNet architecture for 4-corner court detection
    """
    def __init__(self, config):
        super().__init__()
        
        c_band = config['model']['ofe']['channels_per_band']
        c_feat = config['model']['feature_dim']
        num_kpt = config['model']['num_keypoints']
        
        # 1. Octave Feature Extractor (Backbone)
        self.backbone = OctaveFeatureExtractor(c_band=c_band)
        
        # 2. Polar Transform Attention
        self.pta = PolarTransformAttention(
            c=c_band * 2,  # OFE outputs 2*c_band
            r_bins=config['model']['pta']['radial_bins'],
            theta_bins=config['model']['pta']['angular_bins']
        )
        
        # 3. Neck (feature projection)
        self.neck = nn.Sequential(
            nn.Conv2d(c_band * 2, c_feat, 1, bias=False),
            nn.BatchNorm2d(c_feat),
            nn.SiLU(inplace=True)
        )
        
        # 4. Heatmap Head (coarse localization)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(c_feat, c_feat, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_feat),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_feat, num_kpt, 1)
        )
        
        # 5. Keypoint Feature Extractor
        self.kpt_proj = nn.Linear(c_feat, c_feat)
        
        # 6. Transformer Encoder (contextual refinement)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c_feat,
            nhead=config['model']['transformer']['num_heads'],
            dim_feedforward=config['model']['transformer']['dim_feedforward'],
            dropout=config['model']['transformer']['dropout'],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['model']['transformer']['num_layers']
        )
        
        # 7. Quadrilateral Constraint Module
        self.qcm = QuadrilateralConstraintModule(
            feature_dim=c_feat,
            hidden_dims=config['model']['qcm']['hidden_dims']
        )
        
        # 8. Offset Regression Head (bounded with tanh to prevent exploding predictions)
        self.offset_head = nn.Sequential(
            nn.Linear(c_feat, c_feat // 2),
            nn.ReLU(inplace=True),
            nn.Linear(c_feat // 2, 2),
            nn.Tanh()  # Bound to [-1, 1]
        )
        
        # Max offset magnitude (prevents large jumps from initial keypoints)
        self.max_offset = 0.1  # Max 10% of image size

    def soft_argmax_2d(self, heatmaps):
        """
        Differentiable keypoint extraction from heatmaps
        heatmaps: (B, K, H, W)
        Returns: (B, K, 2) normalized coordinates [0, 1]
        """
        B, K, H, W = heatmaps.shape
        device = heatmaps.device
        
        # Flatten and softmax
        hm_flat = heatmaps.view(B, K, -1)
        probs = F.softmax(hm_flat, dim=-1)
        
        # Create coordinate grids
        y = torch.linspace(0, 1, H, device=device)
        x = torch.linspace(0, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)  # (HW, 2)
        
        # Weighted sum (expectation)
        coords = torch.matmul(probs, grid)  # (B, K, 2)
        return coords

    def forward(self, x):
        """
        x: (B, 3, H, W) input image
        Returns dict with:
          - heatmaps: (B, 4, H', W')
          - kpts_init: (B, 4, 2) initial coordinates
          - kpts_refined: (B, 4, 2) final refined coordinates
        """
        # 1. Feature extraction
        feat = self.backbone(x)        # (B, 128, H/2, W/2)
        feat = self.pta(feat)          # (B, 128, H/2, W/2)
        feat = self.neck(feat)         # (B, C, H/2, W/2)
        
        # 2. Heatmap prediction
        heatmaps = self.heatmap_head(feat)  # (B, 4, H/2, W/2)
        
        # 3. Initial keypoint extraction (soft-argmax)
        kpts_init = self.soft_argmax_2d(heatmaps)  # (B, 4, 2)
        
        # 4. Sample features at keypoint locations
        grid = kpts_init.view(x.size(0), 4, 1, 2) * 2 - 1  # → [-1, 1]
        kpt_feats = F.grid_sample(
            feat, grid, align_corners=False
        ).squeeze(-1).permute(0, 2, 1)  # (B, 4, C)
        
        # 5. Project and refine with transformer
        kpt_feats = self.kpt_proj(kpt_feats)
        kpt_feats = self.transformer(kpt_feats)
        
        # 6. Apply geometric constraints
        kpt_feats = self.qcm(kpt_feats, kpts_init)
        
        # 7. Predict refinement offsets (bounded)
        offsets = self.offset_head(kpt_feats) * self.max_offset  # (B, 4, 2), scaled to [-0.1, 0.1]
        
        # 8. Final refined keypoints
        kpts_refined = torch.clamp(kpts_init + offsets, 0, 1)
        
        return {
            'heatmaps': heatmaps,
            'kpts_init': kpts_init,
            'kpts_refined': kpts_refined,
            'offsets': offsets
        }
