"""STEAR-style latent-memory intervention for LatentMAS.

The original STEAR method is an inference-time intervention that triggers on
uncertain decoding steps, selects token-conditioned evidence, reinjects that
evidence in a grounding-sensitive representation region, and optionally uses a
counterfactual branch for contrastive calibration.

LatentMAS does not expose visual patches in its text-only latent reasoning
path, so this module maps the same mechanics onto the shared latent working
memory:

* high-risk step detection uses next-token logit uncertainty when logits are
  available;
* key evidence selection retrieves the most query-relevant latent memory slots;
* middle-layer evidence reinjection becomes a residual update to the active
  latent state or memory slot;
* temporal counterfactuals perturb selected latent-memory positions, preserving
  the rest of the agent collaboration trace.

The implementation is intentionally backend-light. It works with NumPy arrays
for tests and with PyTorch tensors in an upstream LatentMAS runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Literal, Mapping, Optional, Tuple

ArrayLike = Any
CounterfactualMode = Literal["reverse", "shuffle", "homogenize"]


@dataclass(frozen=True)
class STEARConfig:
    """Configuration for STEAR latent-memory interventions."""

    enabled: bool = False
    trigger_threshold: float = 0.65
    margin_threshold: float = 0.05
    evidence_ratio: float = 0.25
    min_evidence_tokens: int = 1
    max_evidence_tokens: int = 16
    injection_strength: float = 0.15
    counterfactual_alpha: float = 0.5
    counterfactual_mode: CounterfactualMode = "reverse"
    homogenize_gamma: float = 1.0
    eps: float = 1e-8

    @classmethod
    def from_args(cls, args: Any) -> "STEARConfig":
        """Build config from an argparse namespace without coupling to argparse."""

        return cls(
            enabled=bool(getattr(args, "stear", getattr(args, "enable_stear", False))),
            trigger_threshold=float(getattr(args, "stear_trigger_threshold", cls.trigger_threshold)),
            margin_threshold=float(getattr(args, "stear_margin_threshold", cls.margin_threshold)),
            evidence_ratio=float(getattr(args, "stear_evidence_ratio", cls.evidence_ratio)),
            min_evidence_tokens=int(getattr(args, "stear_min_evidence_tokens", cls.min_evidence_tokens)),
            max_evidence_tokens=int(getattr(args, "stear_max_evidence_tokens", cls.max_evidence_tokens)),
            injection_strength=float(getattr(args, "stear_injection_strength", cls.injection_strength)),
            counterfactual_alpha=float(getattr(args, "stear_counterfactual_alpha", cls.counterfactual_alpha)),
            counterfactual_mode=getattr(args, "stear_counterfactual_mode", cls.counterfactual_mode),
            homogenize_gamma=float(getattr(args, "stear_homogenize_gamma", cls.homogenize_gamma)),
        )

    def as_cli_flags(self) -> Dict[str, Any]:
        """Return a serializable representation useful for experiment logging."""

        return {
            "stear": self.enabled,
            "stear_trigger_threshold": self.trigger_threshold,
            "stear_margin_threshold": self.margin_threshold,
            "stear_evidence_ratio": self.evidence_ratio,
            "stear_min_evidence_tokens": self.min_evidence_tokens,
            "stear_max_evidence_tokens": self.max_evidence_tokens,
            "stear_injection_strength": self.injection_strength,
            "stear_counterfactual_alpha": self.counterfactual_alpha,
            "stear_counterfactual_mode": self.counterfactual_mode,
            "stear_homogenize_gamma": self.homogenize_gamma,
        }


@dataclass(frozen=True)
class STEARDecision:
    """Per-batch intervention metadata."""

    triggered: ArrayLike
    uncertainty: Optional[ArrayLike] = None
    margin: Optional[ArrayLike] = None
    evidence_indices: Optional[ArrayLike] = None
    evidence_scores: Optional[ArrayLike] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class STEARIntervention:
    """Result of applying STEAR to a latent-memory tensor."""

    positive_memory: ArrayLike
    negative_memory: Optional[ArrayLike]
    decision: STEARDecision


class LatentSTEARController:
    """Apply STEAR-inspired intervention to LatentMAS hidden-state memory."""

    def __init__(self, config: Optional[STEARConfig] = None) -> None:
        self.config = config or STEARConfig()

    def uncertainty_from_logits(self, logits: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """Return normalized entropy and top-two probability margin.

        Normalized entropy is in [0, 1] for finite logits. A high entropy or a
        low top-two margin indicates a high-risk decoding step.
        """

        probs = _softmax(logits, axis=-1)
        vocab = int(probs.shape[-1])
        entropy = -_sum(probs * _log(_clip(probs, self.config.eps, 1.0)), axis=-1)
        if vocab > 1:
            entropy = entropy / math.log(vocab)
            top2 = _topk_values(probs, 2, axis=-1)
            margin = top2[..., 0] - top2[..., 1]
        else:
            margin = _zeros_like(entropy)
        return entropy, margin

    def trigger_mask(
        self,
        logits: Optional[ArrayLike] = None,
        uncertainty: Optional[ArrayLike] = None,
        margin: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        """Return a boolean mask identifying high-risk batch elements."""

        if not self.config.enabled:
            ref = logits if logits is not None else uncertainty if uncertainty is not None else margin
            if ref is None:
                raise ValueError("logits, uncertainty, or margin is required")
            return _zeros_bool_like_batch(ref)

        if uncertainty is None or margin is None:
            if logits is None:
                raise ValueError("logits are required when uncertainty and margin are not both provided")
            uncertainty, margin = self.uncertainty_from_logits(logits)

        return (uncertainty >= self.config.trigger_threshold) | (margin <= self.config.margin_threshold)

    def select_key_evidence(
        self,
        memory: ArrayLike,
        query: Optional[ArrayLike] = None,
        attention_scores: Optional[ArrayLike] = None,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Select the most relevant latent-memory slots for each batch item.

        Args:
            memory: Latent memory with shape [batch, memory_len, hidden].
            query: Active hidden state with shape [batch, hidden] or
                [batch, query_len, hidden]. If omitted, the last memory slot is
                used.
            attention_scores: Optional precomputed routing scores with shape
                [batch, memory_len]. This can be supplied by a model that
                exposes layer attention.
        """

        _require_rank(memory, 3, "memory")
        batch_size, memory_len, _ = memory.shape
        if memory_len == 0:
            raise ValueError("memory must contain at least one latent slot")

        k = self._evidence_count(memory_len)
        if attention_scores is not None:
            scores = attention_scores
        else:
            if query is None:
                query = memory[:, -1, :]
            query = _last_query(query)
            scores = _cosine_similarity(memory, query, eps=self.config.eps)

        values, indices = _topk(scores, k, axis=-1)
        return _reshape_indices(indices, batch_size, k), values

    def reinject(self, target: ArrayLike, evidence: ArrayLike) -> ArrayLike:
        """Residual-inject selected evidence into the active latent state."""

        _require_rank(evidence, 3, "evidence")
        target_rank = _rank(target)
        target_2d = _last_query(target)
        evidence_summary = _mean(evidence, axis=1)
        residual = _normalize(evidence_summary, eps=self.config.eps)
        target_norm = _l2_norm(target_2d, axis=-1, keepdims=True, eps=self.config.eps)
        residual = residual * target_norm
        updated = target_2d + self.config.injection_strength * residual
        if target_rank == 3:
            result = _copy(target)
            result[:, -1, :] = updated
            return result
        if target_rank == 2:
            return updated
        raise ValueError("target must have shape [batch, hidden] or [batch, length, hidden]")

    def build_counterfactual_memory(self, memory: ArrayLike, indices: ArrayLike) -> ArrayLike:
        """Perturb only selected latent-memory positions for contrastive decoding."""

        _require_rank(memory, 3, "memory")
        mode = self.config.counterfactual_mode
        if mode not in {"reverse", "shuffle", "homogenize"}:
            raise ValueError(f"Unsupported counterfactual_mode: {mode}")

        result = _copy(memory)
        batch_size, _, _ = memory.shape
        for batch_idx in range(batch_size):
            selected = _indices_to_list(indices[batch_idx])
            if len(selected) < 2 and mode in {"reverse", "shuffle"}:
                continue
            if mode == "homogenize":
                full_mean = _mean(memory[batch_idx : batch_idx + 1, :, :], axis=1)[0]
                gamma = self.config.homogenize_gamma
                for idx in selected:
                    result[batch_idx, idx, :] = (1.0 - gamma) * result[batch_idx, idx, :] + gamma * full_mean
            else:
                replacement = list(reversed(selected))
                for src_idx, dst_idx in zip(replacement, selected):
                    result[batch_idx, dst_idx, :] = memory[batch_idx, src_idx, :]
        return result

    def contrastive_logits(self, positive_logits: ArrayLike, negative_logits: ArrayLike) -> ArrayLike:
        """Apply STEAR-style positive/negative branch logit calibration."""

        alpha = self.config.counterfactual_alpha
        return (1.0 + alpha) * positive_logits - alpha * negative_logits

    def intervene_latent_memory(
        self,
        memory: ArrayLike,
        *,
        query: Optional[ArrayLike] = None,
        logits: Optional[ArrayLike] = None,
        uncertainty: Optional[ArrayLike] = None,
        margin: Optional[ArrayLike] = None,
        attention_scores: Optional[ArrayLike] = None,
    ) -> STEARIntervention:
        """Run the full STEAR latent-memory intervention.

        When disabled or when no batch element triggers, this returns the
        original memory and no negative branch. Triggered batch elements receive
        the reinjected memory slot and a localized counterfactual memory.
        """

        _require_rank(memory, 3, "memory")
        if not self.config.enabled:
            mask = self.trigger_mask(logits=logits if logits is not None else memory[:, 0, 0])
            return STEARIntervention(memory, None, STEARDecision(triggered=mask))

        if uncertainty is None or margin is None:
            if logits is not None:
                uncertainty, margin = self.uncertainty_from_logits(logits)
            else:
                ref = memory[:, 0, 0]
                uncertainty = _ones_like(ref)
                margin = _zeros_like(ref)

        mask = self.trigger_mask(uncertainty=uncertainty, margin=margin)
        indices, scores = self.select_key_evidence(memory, query=query, attention_scores=attention_scores)
        evidence = _gather_memory(memory, indices)
        reinjected_query = self.reinject(query if query is not None else memory[:, -1, :], evidence)

        positive_memory = _copy(memory)
        if _rank(reinjected_query) == 3:
            candidate = reinjected_query[:, -1, :]
        else:
            candidate = reinjected_query
        positive_memory = _where_batch(mask, _replace_last(positive_memory, candidate), positive_memory)

        negative_candidate = self.build_counterfactual_memory(memory, indices)
        negative_memory = _where_batch(mask, negative_candidate, memory)
        if not _any(mask):
            negative_memory = None

        decision = STEARDecision(
            triggered=mask,
            uncertainty=uncertainty,
            margin=margin,
            evidence_indices=indices,
            evidence_scores=scores,
            metadata={
                "evidence_ratio": self.config.evidence_ratio,
                "injection_strength": self.config.injection_strength,
                "counterfactual_mode": self.config.counterfactual_mode,
            },
        )
        return STEARIntervention(positive_memory, negative_memory, decision)

    def _evidence_count(self, memory_len: int) -> int:
        raw = int(round(memory_len * self.config.evidence_ratio))
        count = max(self.config.min_evidence_tokens, raw)
        count = min(self.config.max_evidence_tokens, count)
        return max(1, min(memory_len, count))


def _array_namespace(x: ArrayLike) -> Any:
    module = type(x).__module__.split(".")[0]
    if module == "torch":
        import torch

        return torch
    import numpy as np

    return np


def _rank(x: ArrayLike) -> int:
    return len(x.shape)


def _require_rank(x: ArrayLike, rank: int, name: str) -> None:
    if _rank(x) != rank:
        raise ValueError(f"{name} must have rank {rank}; got shape {getattr(x, 'shape', None)}")


def _batch_size(x: ArrayLike) -> int:
    if _rank(x) == 0:
        return 1
    return int(x.shape[0])


def _copy(x: ArrayLike) -> ArrayLike:
    xp = _array_namespace(x)
    if xp.__name__ == "torch":
        return x.clone()
    return x.copy()


def _softmax(x: ArrayLike, axis: int = -1) -> ArrayLike:
    xp = _array_namespace(x)
    if xp.__name__ == "torch":
        return xp.softmax(x, dim=axis)
    shifted = x - xp.max(x, axis=axis, keepdims=True)
    exp = xp.exp(shifted)
    return exp / xp.sum(exp, axis=axis, keepdims=True)


def _topk(x: ArrayLike, k: int, axis: int = -1) -> Tuple[ArrayLike, ArrayLike]:
    xp = _array_namespace(x)
    if xp.__name__ == "torch":
        return xp.topk(x, k, dim=axis)
    import numpy as np

    indices = np.argsort(x, axis=axis)[..., -k:][..., ::-1]
    values = np.take_along_axis(x, indices, axis=axis)
    return values, indices


def _topk_values(x: ArrayLike, k: int, axis: int = -1) -> ArrayLike:
    values, _ = _topk(x, k, axis=axis)
    return values


def _sum(x: ArrayLike, axis: int = -1) -> ArrayLike:
    xp = _array_namespace(x)
    if xp.__name__ == "torch":
        return xp.sum(x, dim=axis)
    return xp.sum(x, axis=axis)


def _mean(x: ArrayLike, axis: int = -1) -> ArrayLike:
    xp = _array_namespace(x)
    if xp.__name__ == "torch":
        return xp.mean(x, dim=axis)
    return xp.mean(x, axis=axis)


def _log(x: ArrayLike) -> ArrayLike:
    xp = _array_namespace(x)
    return xp.log(x)


def _clip(x: ArrayLike, minimum: float, maximum: float) -> ArrayLike:
    xp = _array_namespace(x)
    if xp.__name__ == "torch":
        return xp.clamp(x, min=minimum, max=maximum)
    return xp.clip(x, minimum, maximum)


def _zeros_like(x: ArrayLike) -> ArrayLike:
    xp = _array_namespace(x)
    return xp.zeros_like(x)


def _ones_like(x: ArrayLike) -> ArrayLike:
    xp = _array_namespace(x)
    return xp.ones_like(x)


def _zeros_bool_like_batch(x: ArrayLike) -> ArrayLike:
    xp = _array_namespace(x)
    if xp.__name__ == "torch":
        return x.new_zeros((_batch_size(x),), dtype=xp.bool)
    return xp.zeros(_batch_size(x), dtype=bool)


def _l2_norm(x: ArrayLike, axis: int = -1, keepdims: bool = True, eps: float = 1e-8) -> ArrayLike:
    xp = _array_namespace(x)
    if xp.__name__ == "torch":
        return xp.linalg.vector_norm(x, dim=axis, keepdim=keepdims).clamp_min(eps)
    return xp.maximum(xp.linalg.norm(x, axis=axis, keepdims=keepdims), eps)


def _normalize(x: ArrayLike, eps: float = 1e-8) -> ArrayLike:
    return x / _l2_norm(x, axis=-1, keepdims=True, eps=eps)


def _cosine_similarity(memory: ArrayLike, query: ArrayLike, eps: float) -> ArrayLike:
    query = _normalize(query, eps=eps)
    memory = _normalize(memory, eps=eps)
    xp = _array_namespace(memory)
    if xp.__name__ == "torch":
        return xp.sum(memory * query[:, None, :], dim=-1)
    return xp.sum(memory * query[:, None, :], axis=-1)


def _last_query(query: ArrayLike) -> ArrayLike:
    if _rank(query) == 3:
        return query[:, -1, :]
    if _rank(query) == 2:
        return query
    raise ValueError("query must have shape [batch, hidden] or [batch, length, hidden]")


def _reshape_indices(indices: ArrayLike, batch_size: int, k: int) -> ArrayLike:
    if tuple(indices.shape) == (batch_size, k):
        return indices
    xp = _array_namespace(indices)
    if xp.__name__ == "torch":
        return indices.reshape(batch_size, k)
    return indices.reshape(batch_size, k)


def _indices_to_list(indices: ArrayLike) -> Iterable[int]:
    if hasattr(indices, "detach"):
        indices = indices.detach().cpu().tolist()
    elif hasattr(indices, "tolist"):
        indices = indices.tolist()
    return [int(idx) for idx in indices]


def _gather_memory(memory: ArrayLike, indices: ArrayLike) -> ArrayLike:
    xp = _array_namespace(memory)
    if xp.__name__ == "torch":
        expanded = indices[..., None].expand(-1, -1, memory.shape[-1])
        return memory.gather(dim=1, index=expanded)

    import numpy as np

    return np.take_along_axis(memory, indices[..., None], axis=1)


def _replace_last(memory: ArrayLike, replacement: ArrayLike) -> ArrayLike:
    result = _copy(memory)
    result[:, -1, :] = replacement
    return result


def _where_batch(mask: ArrayLike, yes: ArrayLike, no: ArrayLike) -> ArrayLike:
    xp = _array_namespace(yes)
    if xp.__name__ == "torch":
        view_shape = [int(mask.shape[0])] + [1] * (len(yes.shape) - 1)
        return xp.where(mask.reshape(view_shape), yes, no)
    view_shape = (int(mask.shape[0]),) + (1,) * (len(yes.shape) - 1)
    return xp.where(mask.reshape(view_shape), yes, no)


def _any(mask: ArrayLike) -> bool:
    xp = _array_namespace(mask)
    if xp.__name__ == "torch":
        return bool(mask.any().item())
    return bool(xp.any(mask))
