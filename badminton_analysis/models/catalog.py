"""Single source of truth for the shuttlecock model layout."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    framework: str
    architecture_module: str
    default_weights: str


MODEL_CATALOG: Dict[str, ModelSpec] = {
    "tracknet_v4": ModelSpec(
        key="tracknet_v4",
        display_name="TrackNetV4 Type B",
        framework="Keras 3 (PyTorch backend)",
        architecture_module="badminton_analysis.models.tracknet_v4.architecture",
        default_weights="weights/tracknet_v4/tracknet_v4_type_b.keras",
    ),
    "tracknet": ModelSpec(
        key="tracknet",
        display_name="TrackNetV3",
        framework="PyTorch",
        architecture_module="badminton_analysis.models.tracknet_v3.architecture",
        default_weights="weights/tracknet/tracknet_v3_tracker.pt",
    ),
    "yolo": ModelSpec(
        key="yolo",
        display_name="YOLO11 shuttlecock",
        framework="Ultralytics/PyTorch",
        architecture_module="badminton_analysis.models.yolo11.runtime",
        default_weights="weights/yolo11s-ball.pt",
    ),
}


def get_model_spec(key: str) -> ModelSpec:
    try:
        return MODEL_CATALOG[key]
    except KeyError as exc:
        supported = ", ".join(MODEL_CATALOG)
        raise ValueError(f"Unknown model {key!r}; expected one of: {supported}") from exc

