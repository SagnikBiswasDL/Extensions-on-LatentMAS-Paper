"""Integration helpers for SEAL-like STEAR in LatentMAS."""

from __future__ import annotations

from typing import Any, Optional

from .core import LatentSTEARController, STEARConfig, STEARIntervention, load_steering_vector


def add_stear_arguments(parser: Any) -> Any:
    """Register CLI flags for latent reasoning steering."""

    parser.add_argument("--stear", action="store_true", help="Enable SEAL-like latent reasoning steering.")
    parser.add_argument(
        "--stear_vector_path",
        type=str,
        default=None,
        help="Path to a .npy/.npz steering vector extracted from labeled thought hidden states.",
    )
    parser.add_argument(
        "--stear_alpha",
        type=float,
        default=STEARConfig.alpha,
        help="Strength for hidden-state intervention: H <- H + alpha * S.",
    )
    parser.add_argument(
        "--stear_boundary_strategy",
        choices=["last", "all", "indices"],
        default=STEARConfig.boundary_strategy,
        help="Which latent memory slots to steer. 'last' matches the judger handoff use case.",
    )
    parser.add_argument(
        "--stear_normalize_vector",
        action="store_true",
        default=STEARConfig.normalize_vector,
        help="Normalize extracted steering vectors before applying them.",
    )
    parser.add_argument(
        "--stear_no_normalize_vector",
        action="store_false",
        dest="stear_normalize_vector",
        help="Use the raw extracted steering vector magnitude.",
    )
    parser.add_argument(
        "--stear_preserve_hidden_norm",
        action="store_true",
        help="Rescale steered states back to their original hidden-state norm.",
    )
    return parser


def build_controller_from_args(args: Any) -> LatentSTEARController:
    """Create a controller and load a vector when ``--stear_vector_path`` is set."""

    config = STEARConfig.from_args(args)
    vector = load_steering_vector(config.steering_vector_path) if config.steering_vector_path else None
    return LatentSTEARController(config=config, steering_vector=vector)


def apply_stear_to_latent_memory(
    past_embedding: Any,
    *,
    controller: Optional[LatentSTEARController] = None,
    args: Optional[Any] = None,
    steering_vector: Optional[Any] = None,
    boundary_indices: Optional[Any] = None,
) -> STEARIntervention:
    """Apply SEAL-like steering to LatentMAS ``embedding_record`` memory.

    In upstream LatentMAS, call this after:

    ``past_embedding = torch.cat(embedding_record, dim=1).to(self.vllm_device)``

    Then replace ``past_embedding`` with ``intervention.steered_memory`` before
    inserting latent memory into the judger prompt.
    """

    if controller is None:
        controller = build_controller_from_args(args) if args is not None else LatentSTEARController()
    return controller.apply_to_memory(
        past_embedding,
        steering_vector=steering_vector,
        boundary_indices=boundary_indices,
    )
