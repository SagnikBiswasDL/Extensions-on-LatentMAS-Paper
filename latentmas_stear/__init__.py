"""SEAL-like STEAR utilities for LatentMAS reasoning experiments."""

from .core import (
    DEFAULT_REFLECTION_KEYWORDS,
    DEFAULT_TRANSITION_KEYWORDS,
    LatentSTEARController,
    STEARConfig,
    STEARDecision,
    STEARIntervention,
    SteeringVector,
    classify_reasoning_trace,
    classify_thought,
    compute_reasoning_steering_vector,
    load_steering_vector,
    save_steering_vector,
    split_reasoning_trace,
)
from .integration import add_stear_arguments, apply_stear_to_latent_memory, build_controller_from_args

__all__ = [
    "DEFAULT_REFLECTION_KEYWORDS",
    "DEFAULT_TRANSITION_KEYWORDS",
    "LatentSTEARController",
    "STEARConfig",
    "STEARDecision",
    "STEARIntervention",
    "SteeringVector",
    "add_stear_arguments",
    "apply_stear_to_latent_memory",
    "build_controller_from_args",
    "classify_reasoning_trace",
    "classify_thought",
    "compute_reasoning_steering_vector",
    "load_steering_vector",
    "save_steering_vector",
    "split_reasoning_trace",
]
