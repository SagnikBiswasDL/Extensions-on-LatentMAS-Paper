"""Convenience hooks for wiring STEAR into LatentMAS experiments."""

from __future__ import annotations

from typing import Any, Optional

from .core import LatentSTEARController, STEARConfig, STEARIntervention


def add_stear_arguments(parser: Any) -> Any:
    """Register CLI flags expected by :class:`STEARConfig.from_args`.

    This function accepts an ``argparse.ArgumentParser``-compatible object and
    returns it for fluent setup in upstream ``run.py`` files.
    """

    parser.add_argument("--stear", action="store_true", help="Enable STEAR latent-memory intervention for LatentMAS.")
    parser.add_argument(
        "--stear_trigger_threshold",
        type=float,
        default=STEARConfig.trigger_threshold,
        help="Normalized entropy threshold above which STEAR intervenes.",
    )
    parser.add_argument(
        "--stear_margin_threshold",
        type=float,
        default=STEARConfig.margin_threshold,
        help="Top-two probability margin below which STEAR intervenes.",
    )
    parser.add_argument(
        "--stear_evidence_ratio",
        type=float,
        default=STEARConfig.evidence_ratio,
        help="Fraction of latent-memory slots selected as key evidence.",
    )
    parser.add_argument(
        "--stear_min_evidence_tokens",
        type=int,
        default=STEARConfig.min_evidence_tokens,
        help="Minimum number of latent slots selected as evidence.",
    )
    parser.add_argument(
        "--stear_max_evidence_tokens",
        type=int,
        default=STEARConfig.max_evidence_tokens,
        help="Maximum number of latent slots selected as evidence.",
    )
    parser.add_argument(
        "--stear_injection_strength",
        type=float,
        default=STEARConfig.injection_strength,
        help="Residual strength for injecting selected evidence into the active latent state.",
    )
    parser.add_argument(
        "--stear_counterfactual_alpha",
        type=float,
        default=STEARConfig.counterfactual_alpha,
        help="Contrastive logit strength for positive-vs-counterfactual calibration.",
    )
    parser.add_argument(
        "--stear_counterfactual_mode",
        choices=["reverse", "shuffle", "homogenize"],
        default=STEARConfig.counterfactual_mode,
        help="How selected latent slots are perturbed in the counterfactual branch.",
    )
    parser.add_argument(
        "--stear_homogenize_gamma",
        type=float,
        default=STEARConfig.homogenize_gamma,
        help="Interpolation strength for the homogenize counterfactual mode.",
    )
    return parser


def build_controller_from_args(args: Any) -> LatentSTEARController:
    """Create a STEAR controller from a LatentMAS argparse namespace."""

    return LatentSTEARController(STEARConfig.from_args(args))


def apply_stear_to_latent_memory(
    past_embedding: Any,
    *,
    controller: Optional[LatentSTEARController] = None,
    args: Optional[Any] = None,
    query: Optional[Any] = None,
    logits: Optional[Any] = None,
    uncertainty: Optional[Any] = None,
    margin: Optional[Any] = None,
    attention_scores: Optional[Any] = None,
) -> STEARIntervention:
    """Apply STEAR to the LatentMAS embedding record.

    In upstream LatentMAS, ``past_embedding`` corresponds to the concatenated
    ``embedding_record`` tensor in ``LatentMASMethod.run_batch_vllm`` with shape
    ``[batch, latent_memory_len, hidden]``. The returned ``positive_memory`` can
    replace that tensor before it is inserted into the judger prompt embedding.

    If the caller can run a second counterfactual decode, ``negative_memory`` is
    the matching perturbed memory for that branch; otherwise it can be ignored
    and the positive reinjection still provides the main STEAR intervention.
    """

    if controller is None:
        controller = build_controller_from_args(args) if args is not None else LatentSTEARController()
    return controller.intervene_latent_memory(
        past_embedding,
        query=query,
        logits=logits,
        uncertainty=uncertainty,
        margin=margin,
        attention_scores=attention_scores,
    )
