"""
Polar Transform Attention
Manuscript: Section 3.3, Equations (19)-(25)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PolarTransformAttention(nn.Module):
    """
    Converts features to polar coordinates, processes, then generates attention
    """
    def __init__(self, c, r_bins=64, theta_bins=128):
        super().__init__()
        self.r_bins = r_bins
        self.theta_bins = theta_bins
        
        # Polar domain convolution
        self.polar_conv = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c)
        )
        
        # Attention generation
        self.attn_head = nn.Sequential(
            nn.Conv2d(c, c // 4, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c // 4, 1, 1),
            nn.Sigmoid()
        )

    def cartesian_to_polar_grid(self, H, W, device):
        """
        Create sampling grid for Cartesian → Polar transformation
        Returns grid in range [-1, 1] for F.grid_sample
        """
        # Polar coordinates
        theta = torch.linspace(0, 2*math.pi, self.theta_bins, device=device)
        r = torch.linspace(0, 1, self.r_bins, device=device)
        
        # Meshgrid
        R, THETA = torch.meshgrid(r, theta, indexing='ij')
        
        # Convert to Cartesian (centered at image center)
        X = R * torch.cos(THETA)  # [-1, 1]
        Y = R * torch.sin(THETA)  # [-1, 1]
        
        # Stack to grid shape (r_bins, theta_bins, 2)
        grid = torch.stack([X, Y], dim=-1)
        return grid

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        
        # 1. Cartesian → Polar transformation
        polar_grid = self.cartesian_to_polar_grid(H, W, device)
        polar_grid = polar_grid.unsqueeze(0).expand(B, -1, -1, -1)
        
        x_polar = F.grid_sample(x, polar_grid, align_corners=False)
        
        # 2. Process in polar domain
        x_polar = self.polar_conv(x_polar)
        
        # 3. Interpolate back to Cartesian size
        x_polar_cart = F.interpolate(x_polar, size=(H, W), 
                                     mode='bilinear', align_corners=False)
        
        # 4. Generate attention map
        attn = self.attn_head(x_polar_cart)
        
        # 5. Apply attention
        return x * attn
