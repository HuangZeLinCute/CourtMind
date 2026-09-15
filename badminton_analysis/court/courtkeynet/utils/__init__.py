# -*- coding: utf-8 -*-
"""Minimal ``utils`` package for the vendored CourtKeyNet inference path.

The original CourtKeyNet ``utils/__init__.py`` imports ``dataloader`` and
``metrics``, which require ``albumentations`` / ``shapely`` — dependencies that
are intentionally NOT installed in Good-Badminton (they conflict with the
project's OpenCV 4.10 / numpy<2 stack). Only the safetensors weight I/O is
needed for inference, so only that is exposed here.
"""
from ._safetensors import load_weights

__all__ = ["load_weights"]
