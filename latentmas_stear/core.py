"""SEAL-like steering calibration for LatentMAS reasoning.

This module implements the version of "STEAR" meant in the SEAL sense:
training-free steerable reasoning calibration. The core idea is to extract a
latent steering vector that points from redundant reasoning modes
(reflection/transition) toward direct execution reasoning, then add that vector
to hidden states during inference.

For LatentMAS, the intervention target is the shared latent working memory
rather than visible chain-of-thought tokens. Agent handoffs and latent rollout
slots act as the reasoning boundaries where the steering vector can calibrate
the next reasoning step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

ArrayLike = Any
ThoughtType = Literal["execution", "reflection", "transition"]
BoundaryStrategy = Literal["last", "all", "indices"]


DEFAULT_REFLECTION_KEYWORDS: Tuple[str, ...] = (
    "check",
    "double-check",
    "verify",
    "verification",
    "recheck",
    "make sure",
    "wait",
    "hold on",
    "let me see",
    "is this correct",
    "mistake",
    "error",
    "wrong",
    "reconsider",
)

DEFAULT_TRANSITION_KEYWORDS: Tuple[str, ...] = (
    "alternatively",
    "another approach",
    "another way",
    "different approach",
    "instead",
    "new approach",
    "try a different",
    "let's try",
    "let me try",
    "switch",
    "restart",
    "from another perspective",
)


@dataclass(frozen=True)
class STEARConfig:
    """Configuration for SEAL-like latent steering."""

    enabled: bool = False
    alpha: float = 1.0
    boundary_strategy: BoundaryStrategy = "last"
    normalize_vector: bool = True
    preserve_hidden_norm: bool = False
    steering_vector_path: Optional[str] = None
    reflection_keywords: Tuple[str, ...] = DEFAULT_REFLECTION_KEYWORDS
    transition_keywords: Tuple[str, ...] = DEFAULT_TRANSITION_KEYWORDS
    eps: float = 1e-8

    @classmethod
    def from_args(cls, args: Any) -> "STEARConfig":
        """Build config from an argparse namespace without importing argparse."""

        return cls(
            enabled=bool(getattr(args, "stear", getattr(args, "enable_stear", False))),
            alpha=float(getattr(args, "stear_alpha", getattr(args, "stear_strength", cls.alpha))),
            boundary_strategy=getattr(args, "stear_boundary_strategy", cls.boundary_strategy),
            normalize_vector=bool(getattr(args, "stear_normalize_vector", cls.normalize_vector)),
            preserve_hidden_norm=bool(getattr(args, "stear_preserve_hidden_norm", cls.preserve_hidden_norm)),
            steering_vector_path=getattr(args, "stear_vector_path", cls.steering_vector_path),
        )

    def as_cli_flags(self) -> Dict[str, Any]:
        """Return a serializable representation for experiment logging."""

        return {
            "stear": self.enabled,
            "stear_alpha": self.alpha,
            "stear_boundary_strategy": self.boundary_strategy,
            "stear_normalize_vector": self.normalize_vector,
            "stear_preserve_hidden_norm": self.preserve_hidden_norm,
            "stear_vector_path": self.steering_vector_path,
        }


@dataclass(frozen=True)
class SteeringVector:
    """Reasoning steering vector and extraction metadata."""

    vector: ArrayLike
    execution_count: int
    reflection_transition_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class STEARDecision:
    """Per-intervention metadata."""

    applied: bool
    boundary_indices: Optional[ArrayLike] = None
    alpha: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class STEARIntervention:
    """Result of applying latent steering."""

    steered_memory: ArrayLike
    decision: STEARDecision


def split_reasoning_trace(text: str, delimiter: str = "\n\n") -> List[str]:
    """Split a generated reasoning trace into thought blocks."""

    return [chunk.strip() for chunk in text.split(delimiter) if chunk.strip()]


def classify_thought(
    thought: str,
    *,
    reflection_keywords: Sequence[str] = DEFAULT_REFLECTION_KEYWORDS,
    transition_keywords: Sequence[str] = DEFAULT_TRANSITION_KEYWORDS,
) -> ThoughtType:
    """Classify a thought block using SEAL-style keyword rules."""

    lowered = thought.lower()
    if any(keyword.lower() in lowered for keyword in transition_keywords):
        return "transition"
    if any(keyword.lower() in lowered for keyword in reflection_keywords):
        return "reflection"
    return "execution"


def classify_reasoning_trace(
    text: str,
    *,
    delimiter: str = "\n\n",
    reflection_keywords: Sequence[str] = DEFAULT_REFLECTION_KEYWORDS,
    transition_keywords: Sequence[str] = DEFAULT_TRANSITION_KEYWORDS,
) -> List[Tuple[str, ThoughtType]]:
    """Split a trace and classify each thought block."""

    return [
        (
            thought,
            classify_thought(
                thought,
                reflection_keywords=reflection_keywords,
                transition_keywords=transition_keywords,
            ),
        )
        for thought in split_reasoning_trace(text, delimiter=delimiter)
    ]


def compute_reasoning_steering_vector(
    representations: ArrayLike,
    labels: Sequence[ThoughtType],
    *,
    normalize: bool = True,
    eps: float = 1e-8,
    metadata: Optional[Mapping[str, Any]] = None,
) -> SteeringVector:
    """Compute the SEAL steering vector ``mean(execution) - mean(reflection/transition)``.

    Args:
        representations: Thought-boundary hidden states with shape
            ``[num_thoughts, hidden]``.
        labels: One label per representation.
        normalize: Whether to L2-normalize the resulting direction.
    """

    _require_rank(representations, 2, "representations")
    if len(labels) != int(representations.shape[0]):
        raise ValueError("labels length must match representations.shape[0]")

    execution_mask = _label_mask(labels, {"execution"}, representations)
    rt_mask = _label_mask(labels, {"reflection", "transition"}, representations)
    execution_count = _scalar_int(_sum(execution_mask, axis=0))
    rt_count = _scalar_int(_sum(rt_mask, axis=0))
    if execution_count == 0:
        raise ValueError("at least one execution representation is required")
    if rt_count == 0:
        raise ValueError("at least one reflection or transition representation is required")

    execution_mean = _masked_mean(representations, execution_mask)
    rt_mean = _masked_mean(representations, rt_mask)
    vector = execution_mean - rt_mean
    if normalize:
        vector = _normalize(vector, eps=eps)

    return SteeringVector(
        vector=vector,
        execution_count=execution_count,
        reflection_transition_count=rt_count,
        metadata=dict(metadata or {}),
    )


def save_steering_vector(path: str | Path, steering_vector: SteeringVector) -> None:
    """Save a steering vector to ``.npy`` or ``.npz`` format."""

    import numpy as np

    destination = Path(path)
    vector = _as_numpy_array(steering_vector.vector)
    if destination.suffix == ".npz":
        np.savez(
            destination,
            vector=vector,
            execution_count=steering_vector.execution_count,
            reflection_transition_count=steering_vector.reflection_transition_count,
        )
    else:
        np.save(destination, vector)


def load_steering_vector(path: str | Path) -> ArrayLike:
    """Load a steering vector from ``.npy`` or ``.npz`` format."""

    import numpy as np

    source = Path(path)
    loaded = np.load(source)
    if hasattr(loaded, "files"):
        return loaded["vector"]
    return loaded


class LatentSTEARController:
    """Apply SEAL-like steering to LatentMAS hidden-state memory."""

    def __init__(
        self,
        config: Optional[STEARConfig] = None,
        steering_vector: Optional[ArrayLike] = None,
    ) -> None:
        self.config = config or STEARConfig()
        self.steering_vector = steering_vector
        if self.steering_vector is None and self.config.steering_vector_path:
            self.steering_vector = load_steering_vector(self.config.steering_vector_path)

    def set_steering_vector(self, steering_vector: ArrayLike | SteeringVector) -> None:
        """Attach or replace the steering vector used at inference time."""

        self.steering_vector = steering_vector.vector if isinstance(steering_vector, SteeringVector) else steering_vector

    def extract_steering_vector(
        self,
        representations: ArrayLike,
        labels: Sequence[ThoughtType],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> SteeringVector:
        """Extract and attach a steering vector from labeled hidden states."""

        steering = compute_reasoning_steering_vector(
            representations,
            labels,
            normalize=self.config.normalize_vector,
            eps=self.config.eps,
            metadata=metadata,
        )
        self.steering_vector = steering.vector
        return steering

    def apply_to_hidden(
        self,
        hidden: ArrayLike,
        *,
        steering_vector: Optional[ArrayLike] = None,
        mask: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        """Apply ``hidden + alpha * steering_vector`` to a hidden-state tensor."""

        vector = self._resolve_vector(hidden, steering_vector)
        if vector is None or not self.config.enabled:
            return hidden

        result = _copy(hidden)
        original_norm = _l2_norm(result, axis=-1, keepdims=True, eps=self.config.eps)
        steered = result + self.config.alpha * _broadcast_vector(vector, result)
        if self.config.preserve_hidden_norm:
            steered = _normalize(steered, eps=self.config.eps) * original_norm
        if mask is not None:
            result = _where_mask(mask, steered, result)
        else:
            result = steered
        return result

    def apply_to_memory(
        self,
        memory: ArrayLike,
        *,
        steering_vector: Optional[ArrayLike] = None,
        boundary_indices: Optional[ArrayLike] = None,
    ) -> STEARIntervention:
        """Apply steering to LatentMAS working memory.

        Args:
            memory: Tensor/array shaped ``[batch, memory_len, hidden]``.
            boundary_indices: Required only when ``boundary_strategy='indices'``.
        """

        _require_rank(memory, 3, "memory")
        vector = self._resolve_vector(memory, steering_vector)
        if vector is None or not self.config.enabled:
            return STEARIntervention(
                memory,
                STEARDecision(
                    applied=False,
                    boundary_indices=boundary_indices,
                    alpha=0.0,
                    metadata={"reason": "disabled_or_missing_vector"},
                ),
            )

        mask = self._boundary_mask(memory, boundary_indices=boundary_indices)
        steered = self.apply_to_hidden(memory, steering_vector=vector, mask=mask)
        return STEARIntervention(
            steered,
            STEARDecision(
                applied=True,
                boundary_indices=boundary_indices,
                alpha=self.config.alpha,
                metadata={"boundary_strategy": self.config.boundary_strategy},
            ),
        )

    def _resolve_vector(self, reference: ArrayLike, steering_vector: Optional[ArrayLike]) -> Optional[ArrayLike]:
        vector = steering_vector if steering_vector is not None else self.steering_vector
        if vector is None:
            return None
        xp = _array_namespace(reference)
        if xp.__name__ == "torch":
            return _to_torch_like(vector, reference)
        return _as_numpy_array(vector).astype(reference.dtype, copy=False)

    def _boundary_mask(self, memory: ArrayLike, *, boundary_indices: Optional[ArrayLike]) -> ArrayLike:
        xp = _array_namespace(memory)
        batch_size, memory_len, _ = memory.shape
        if self.config.boundary_strategy == "all":
            return _ones_bool((batch_size, memory_len, 1), xp, memory)
        if self.config.boundary_strategy == "last":
            mask = _zeros_bool((batch_size, memory_len, 1), xp, memory)
            mask[:, -1, :] = True
            return mask
        if self.config.boundary_strategy != "indices":
            raise ValueError(f"Unsupported boundary_strategy: {self.config.boundary_strategy}")
        if boundary_indices is None:
            raise ValueError("boundary_indices is required when boundary_strategy='indices'")
        mask = _zeros_bool((batch_size, memory_len, 1), xp, memory)
        for batch_idx in range(batch_size):
            for position in _indices_to_list(boundary_indices[batch_idx]):
                if position < 0:
                    position = memory_len + position
                if 0 <= position < memory_len:
                    mask[batch_idx, position, :] = True
        return mask


def _array_namespace(x: ArrayLike) -> Any:
    module = type(x).__module__.split(".")[0]
    if module == "torch":
        import torch

        return torch
    import numpy as np

    return np


def _copy(x: ArrayLike) -> ArrayLike:
    xp = _array_namespace(x)
    if xp.__name__ == "torch":
        return x.clone()
    return x.copy()


def _rank(x: ArrayLike) -> int:
    return len(x.shape)


def _require_rank(x: ArrayLike, rank: int, name: str) -> None:
    if _rank(x) != rank:
        raise ValueError(f"{name} must have rank {rank}; got shape {getattr(x, 'shape', None)}")


def _sum(x: ArrayLike, axis: int = -1) -> ArrayLike:
    xp = _array_namespace(x)
    if xp.__name__ == "torch":
        return xp.sum(x, dim=axis)
    return xp.sum(x, axis=axis)


def _l2_norm(x: ArrayLike, axis: int = -1, keepdims: bool = True, eps: float = 1e-8) -> ArrayLike:
    xp = _array_namespace(x)
    if xp.__name__ == "torch":
        return xp.linalg.vector_norm(x, dim=axis, keepdim=keepdims).clamp_min(eps)
    return xp.maximum(xp.linalg.norm(x, axis=axis, keepdims=keepdims), eps)


def _normalize(x: ArrayLike, eps: float = 1e-8) -> ArrayLike:
    return x / _l2_norm(x, axis=-1, keepdims=True, eps=eps)


def _label_mask(labels: Sequence[ThoughtType], positives: Iterable[str], reference: ArrayLike) -> ArrayLike:
    xp = _array_namespace(reference)
    positive_set = set(positives)
    values = [label in positive_set for label in labels]
    if xp.__name__ == "torch":
        import torch

        return torch.tensor(values, dtype=torch.bool, device=reference.device)
    import numpy as np

    return np.array(values, dtype=bool)


def _masked_mean(values: ArrayLike, mask: ArrayLike) -> ArrayLike:
    selected = values[mask]
    xp = _array_namespace(values)
    if xp.__name__ == "torch":
        return selected.mean(dim=0)
    return selected.mean(axis=0)


def _broadcast_vector(vector: ArrayLike, target: ArrayLike) -> ArrayLike:
    if _rank(vector) == 1:
        return vector.reshape((1,) * (_rank(target) - 1) + (int(vector.shape[-1]),))
    return vector


def _where_mask(mask: ArrayLike, yes: ArrayLike, no: ArrayLike) -> ArrayLike:
    xp = _array_namespace(yes)
    if xp.__name__ == "torch":
        return xp.where(mask, yes, no)
    return xp.where(mask, yes, no)


def _ones_bool(shape: Tuple[int, ...], xp: Any, reference: ArrayLike) -> ArrayLike:
    if xp.__name__ == "torch":
        return reference.new_ones(shape, dtype=xp.bool)
    return xp.ones(shape, dtype=bool)


def _zeros_bool(shape: Tuple[int, ...], xp: Any, reference: ArrayLike) -> ArrayLike:
    if xp.__name__ == "torch":
        return reference.new_zeros(shape, dtype=xp.bool)
    return xp.zeros(shape, dtype=bool)


def _indices_to_list(indices: ArrayLike) -> List[int]:
    if hasattr(indices, "detach"):
        indices = indices.detach().cpu().tolist()
    elif hasattr(indices, "tolist"):
        indices = indices.tolist()
    if isinstance(indices, (int, float)):
        return [int(indices)]
    return [int(idx) for idx in indices]


def _to_numpy(value: ArrayLike) -> ArrayLike:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return value


def _as_numpy_array(value: ArrayLike) -> ArrayLike:
    import numpy as np

    return np.asarray(_to_numpy(value))


def _to_torch_like(value: ArrayLike, reference: ArrayLike) -> ArrayLike:
    import torch

    if hasattr(value, "to") and type(value).__module__.split(".")[0] == "torch":
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _scalar_int(value: ArrayLike) -> int:
    if hasattr(value, "detach"):
        return int(value.detach().cpu().item())
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)
