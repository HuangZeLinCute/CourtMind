# -*- coding: utf-8 -*-
"""CourtKeyNet model package (vendored into Good-Badminton).

This is a self-contained copy of the inference-relevant CourtKeyNet code,
imported from the original repository:
    C:\\Users\\14181\\Desktop\\GraduationProject\\CourtKeyNet-main\\courtkeynet

Only the inference path is vendored (models/, utils/_safetensors.py,
configs/). The original repository is left untouched; keep both in sync
manually if the upstream model architecture changes.
"""
from .models.courtkeynet import CourtKeyNet
from .utils._safetensors import load_weights

__all__ = ["CourtKeyNet", "load_weights"]
