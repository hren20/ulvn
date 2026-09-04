"""Inference wrappers for ViNT and NoMaD local navigation models."""

from .vint_nomad import (
    BaseInferenceTrainer,
    InferenceNoMaDTrainer,
    InferenceViNTTrainer,
)

MODEL_REGISTRY = {
    "nomad": InferenceNoMaDTrainer,
    "vint": InferenceViNTTrainer,
}

__all__ = [
    "BaseInferenceTrainer",
    "InferenceNoMaDTrainer",
    "InferenceViNTTrainer",
    "MODEL_REGISTRY",
]
