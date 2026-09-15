"""TrackNetV3 model architectures reconstructed to match the pretrained
checkpoints shipped with the BadmintonTrackNet project.

The checkpoints in ``BadmintonTrackNet/ckpts`` were produced by a V3 training
run whose code is not part of that repository.  The state dicts are fully
inspectable, however, and reveal:

* ``tracknet_v3_tracker_epoch30.pt`` — a U-Net identical in topology to the
  original TrackNetV3 ``concat`` variant: input is ``(L+1)*3`` channels
  (L RGB frames + 1 median-background RGB frame), output is ``L`` heatmaps
  (one per input frame position).  Every convolution/batchnorm shape matches
  the checkpoint, so loading is strict and deterministic.
* ``tracknet_v3_rectifier_epoch30.pt`` — a 1D U-Net that consumes a per-frame
  4-channel sequence (normalized x, normalized y, inpaint mask, visibility)
  and emits refined normalized coordinates (2 channels).

The naming convention ``encN.M.K`` / ``decN.M.K`` (K=0 conv, K=1 bn) follows the
checkpoint keys exactly.  These classes are written only to be loadable from
those exact checkpoints — they are not a re-implementation of the original
repo's ``tracknet.models.tracknet`` module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Official TrackNetV3 model input resolution.
TRACKNET_HEIGHT = 288
TRACKNET_WIDTH = 512


def _make_conv2d_blocks(in_ch, out_ch, count):
    """Sequential of `count` (Conv2d -> BatchNorm2d -> ReLU) blocks.

    Produces keys like ``enc1.0.0.weight`` (conv) / ``enc1.0.1.weight`` (bn),
    matching the checkpoint naming exactly.
    """
    blocks = []
    for i in range(count):
        cin = in_ch if i == 0 else out_ch
        blocks.append(
            nn.Sequential(
                nn.Conv2d(cin, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        )
    return nn.Sequential(*blocks)


class TrackNetV3Tracker(nn.Module):
    """U-Net tracker matching ``tracknet_v3_tracker_epoch30.pt`` keys.

    Args:
        seq_len: number of input frames (L).  Input channels become (L+1)*3
            because the background median frame is concatenated first.
    """

    def __init__(self, seq_len: int = 8):
        super().__init__()
        self.seq_len = int(seq_len)
        in_ch = (self.seq_len + 1) * 3

        self.backbone = nn.Module()
        self.backbone.enc1 = _make_conv2d_blocks(in_ch, 64, 2)
        self.backbone.enc2 = _make_conv2d_blocks(64, 128, 2)
        self.backbone.enc3 = _make_conv2d_blocks(128, 256, 3)
        self.backbone.enc4 = _make_conv2d_blocks(256, 512, 3)
        self.backbone.dec1 = _make_conv2d_blocks(768, 256, 3)
        self.backbone.dec2 = _make_conv2d_blocks(384, 128, 2)
        self.backbone.dec3 = _make_conv2d_blocks(192, 64, 2)
        self.backbone.output = nn.Conv2d(64, self.seq_len, kernel_size=1)

    def forward(self, x):
        b = self.backbone
        x1 = b.enc1(x)
        x = F.max_pool2d(x1, 2)
        x2 = b.enc2(x)
        x = F.max_pool2d(x2, 2)
        x3 = b.enc3(x)
        x = F.max_pool2d(x3, 2)
        x = b.enc4(x)
        x = torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), x3], dim=1)
        x = b.dec1(x)
        x = torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), x2], dim=1)
        x = b.dec2(x)
        x = torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), x1], dim=1)
        x = b.dec3(x)
        return torch.sigmoid(b.output(x))


def _make_conv1d_blocks(in_ch, out_ch, count):
    """Flat ``count`` x (Conv1d -> LeakyReLU) sequence.

    Produces keys like ``enc1.0.weight`` / ``enc1.2.weight`` (the LeakyReLU has
    no parameters so its index is skipped), matching the checkpoint naming.
    """
    layers = []
    for i in range(count):
        cin = in_ch if i == 0 else out_ch
        layers.append(nn.Conv1d(cin, out_ch, kernel_size=3, padding=1, bias=True))
        layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
    return nn.Sequential(*layers)


class TrackNetV3Rectifier(nn.Module):
    """1D U-Net rectifier matching ``tracknet_v3_rectifier_epoch30.pt`` keys.

    Expects input of shape (B, 4, L) with channels
    [x_norm, y_norm, inpaint_mask, visibility] and returns (B, 2, L)
    refined normalized [x, y].
    """

    def __init__(self, in_ch: int = 4, hidden: int = 64):
        super().__init__()
        self.enc1 = _make_conv1d_blocks(in_ch, hidden, 2)
        self.enc2 = _make_conv1d_blocks(hidden, 128, 2)
        self.enc3 = _make_conv1d_blocks(128, 256, 2)
        self.dec2 = _make_conv1d_blocks(384, 128, 2)
        self.dec1 = _make_conv1d_blocks(192, hidden, 2)
        self.out = nn.Conv1d(hidden, 2, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x = F.max_pool1d(x1, 2)
        x2 = self.enc2(x)
        x = F.max_pool1d(x2, 2)
        x3 = self.enc3(x)
        x = F.interpolate(x3, scale_factor=2, mode="nearest")
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        return self.out(x)


def load_tracknet_v3_tracker(checkpoint_path, seq_len=None, device="cpu"):
    """Load the V3 tracker checkpoint (new BadmintonTrackNet format)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    config = ckpt.get("config", {})
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    seq_len = seq_len or int(model_cfg.get("sequence_length", 8))
    model = TrackNetV3Tracker(seq_len=seq_len)
    model.load_state_dict(sd)
    return model.to(device).eval()


def load_tracknet_v3_rectifier(checkpoint_path, device="cpu"):
    """Load the V3 rectifier checkpoint (new BadmintonTrackNet format)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    model = TrackNetV3Rectifier()
    model.load_state_dict(sd)
    return model.to(device).eval()
