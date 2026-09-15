"""Model implementations bundled with Good-Badminton.

Keep neural-network architectures here and video/pipeline adapters in
``badminton_analysis.shuttlecock``.  This module intentionally avoids importing
Torch, Keras, or Ultralytics so normal API startup stays lightweight.
"""

from .catalog import MODEL_CATALOG, ModelSpec, get_model_spec

__all__ = ["MODEL_CATALOG", "ModelSpec", "get_model_spec"]

