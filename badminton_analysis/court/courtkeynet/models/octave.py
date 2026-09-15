"""
Octave Feature Extractor
Manuscript: Section 3.2, Algorithm 1, Equations (1)-(5)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_silu(c_in, c_out, k=3, s=1, p=1):
    """Standard Conv-BN-SiLU block"""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, s, p, bias=False),
        nn.BatchNorm2d(c_out),
        nn.SiLU(inplace=True)
    )


class CourtSpecificShapeKernels(nn.Module):
    """
    Manuscript: Section 3.2.6, Equations (11)-(18)
    Detects horizontal lines, vertical lines, and corners
    """
    def __init__(self, c_in, c_out=64):
        super().__init__()
        # Horizontal line detector (1×k kernel)
        self.h_conv = nn.Conv2d(c_in, c_out//4, kernel_size=(1, 5), 
                                padding=(0, 2), bias=False)
        # Vertical line detector (k×1 kernel)
        self.v_conv = nn.Conv2d(c_in, c_out//4, kernel_size=(5, 1), 
                                padding=(2, 0), bias=False)
        # Corner detectors (L-shaped patterns)
        self.corner = nn.Conv2d(c_in, c_out//2, kernel_size=3, 
                                padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        h = self.h_conv(x)
        v = self.v_conv(x)
        c = self.corner(x)
        out = torch.cat([h, v, c], dim=1)
        return self.act(self.bn(out))


class FractalResidualBlock(nn.Module):
    """
    Manuscript: Section 3.2.1, Equation (6)
    Simplified fractal block (recursive structure omitted for stability)
    """
    def __init__(self, c):
        super().__init__()
        self.conv1 = conv_bn_silu(c, c, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv2(self.conv1(x)))


class FourierFeatureEncoder(nn.Module):
    """
    Manuscript: Section 3.2.2, Equation (7)
    FFT → Learned Frequency Filter → IFFT
    """
    def __init__(self, c):
        super().__init__()
        # Learnable complex filter (real and imaginary parts)
        self.filter_re = nn.Parameter(torch.randn(1, c, 1, 1) * 0.02)
        self.filter_im = nn.Parameter(torch.randn(1, c, 1, 1) * 0.02)
        self.proj = nn.Conv2d(c * 2, c, 1)

    def forward(self, x):
        # Convert to float32 for FFT stability
        x_f32 = x.float()
        
        # Forward FFT
        X = torch.fft.rfft2(x_f32, norm="ortho")
        
        # Apply learned complex filter (broadcast across spatial frequencies)
        filt = torch.complex(self.filter_re, self.filter_im)
        X_filt = X * filt
        
        # Inverse FFT
        x_freq = torch.fft.irfft2(X_filt, s=x.shape[-2:], norm="ortho")
        
        # Concatenate original + frequency-filtered features
        out = torch.cat([x, x_freq.to(x.dtype)], dim=1)
        return self.proj(out)


class NonLocalSelfSimilarity(nn.Module):
    """
    Manuscript: Section 3.2.4, Equation (9)
    Non-local attention for long-range dependencies
    Memory-efficient version with spatial downsampling
    """
    def __init__(self, c, reduction=2, spatial_reduction=8):
        super().__init__()
        c_red = max(c // reduction, 32)
        self.spatial_reduction = spatial_reduction
        self.q = nn.Conv2d(c, c_red, 1)
        self.k = nn.Conv2d(c, c_red, 1)
        self.v = nn.Conv2d(c, c, 1)
        self.out = nn.Conv2d(c, c, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Spatially downsample for memory efficiency
        if self.spatial_reduction > 1:
            x_down = F.avg_pool2d(x, self.spatial_reduction)
        else:
            x_down = x
        
        _, _, H_d, W_d = x_down.shape
        
        q = self.q(x_down).flatten(2).transpose(1, 2)  # (B, H'd*W'd, C')
        k = self.k(x_down).flatten(2)                   # (B, C', H'd*W'd)
        v = self.v(x_down).flatten(2).transpose(1, 2)   # (B, H'd*W'd, C)
        
        # Scaled dot-product attention
        attn = torch.softmax(q @ k / (q.shape[-1] ** 0.5), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, C, H_d, W_d)
        
        # Upsample back to original resolution
        if self.spatial_reduction > 1:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        
        return x + self.gamma * self.out(out)


class CourtFeatureAggregator(nn.Module):
    """
    Manuscript: Section 3.2.5, Equation (10)
    Multi-orientation filters for view-invariant features
    """
    def __init__(self, c, num_orientations=8):
        super().__init__()
        # Depthwise separable convs with different dilations (orientation proxy)
        self.convs = nn.ModuleList([
            nn.Conv2d(c, c, 3, padding=i+1, dilation=i+1, groups=c, bias=False)
            for i in range(num_orientations)
        ])
        self.fuse = nn.Conv2d(c * num_orientations, c, 1)

    def forward(self, x):
        feats = [conv(x) for conv in self.convs]
        return self.fuse(torch.cat(feats, dim=1))


class OctaveFeatureExtractor(nn.Module):
    """
    Manuscript: Section 3.2, Algorithm 1
    Multi-frequency feature extraction with high/mid/low paths
    """
    def __init__(self, c_band=64):
        super().__init__()
        
        # Stem network
        self.stem = nn.Sequential(
            conv_bn_silu(3, c_band, 3, 2, 1),
            conv_bn_silu(c_band, c_band, 3, 1, 1)
        )
        
        # High-frequency path (fine details)
        self.high_path = nn.Sequential(
            CourtSpecificShapeKernels(c_band, c_band),
            FractalResidualBlock(c_band)
        )
        
        # Mid-frequency path (structural patterns)
        self.mid_path = nn.Sequential(
            CourtSpecificShapeKernels(c_band, c_band),
            FractalResidualBlock(c_band),
            NonLocalSelfSimilarity(c_band)
        )
        
        # Low-frequency path (global context)
        self.low_path = nn.Sequential(
            CourtFeatureAggregator(c_band),
            FourierFeatureEncoder(c_band)
        )
        
        # Fusion layer
        self.fuse = nn.Conv2d(c_band * 3, c_band * 2, 1)

    def forward(self, x):
        # Stem
        x = self.stem(x)  # (B, C, H/2, W/2)
        
        # Split into frequency bands
        f_high = x
        f_mid = F.avg_pool2d(x, 2)  # (B, C, H/4, W/4)
        f_low = F.avg_pool2d(x, 4)  # (B, C, H/8, W/8)
        
        # Process each frequency band
        f_high = self.high_path(f_high)
        f_mid = self.mid_path(f_mid)
        f_low = self.low_path(f_low)
        
        # Upsample to match high-frequency resolution
        f_mid = F.interpolate(f_mid, size=f_high.shape[-2:], 
                             mode='bilinear', align_corners=False)
        f_low = F.interpolate(f_low, size=f_high.shape[-2:], 
                             mode='bilinear', align_corners=False)
        
        # Fuse all frequencies
        fused = self.fuse(torch.cat([f_high, f_mid, f_low], dim=1))
        return fused
