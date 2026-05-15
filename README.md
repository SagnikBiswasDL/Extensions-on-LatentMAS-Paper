# Extensions on LatentMAS: STEAR for Latent Reasoning

This repository contains a lightweight implementation of a **STEAR-style
intervention for LatentMAS**. The goal is not tool calling; it is to test
whether STEAR's inference-time evidence intervention pattern improves latent
multi-agent reasoning.

## What is implemented

The original STEAR method for LLM-style decoding uses four ideas:

1. trigger only on high-risk/uncertain decoding steps;
2. select token-conditioned key evidence;
3. reinject that evidence into the representation region where grounding is
   most useful;
4. optionally calibrate against a localized counterfactual branch.

LatentMAS communicates through hidden-state working memory rather than explicit
visual patches. This extension adapts STEAR to that setting:

- **Uncertainty trigger**: uses normalized next-token entropy and top-two
  probability margin when logits are available.
- **Key evidence selection**: retrieves latent memory slots most aligned with
  the active query hidden state.
- **Latent reinjection**: applies a controlled residual update to the active
  latent state using the selected memory evidence.
- **Counterfactual latent branch**: perturbs only selected latent-memory slots
  (reverse/shuffle or homogenize) for contrastive calibration.

The code is backend-light: it is tested with NumPy and works with PyTorch
tensors in a LatentMAS runtime without importing PyTorch unless tensor inputs
come from PyTorch.

## Files

- `latentmas_stear/core.py` - STEAR controller and tensor operations.
- `latentmas_stear/integration.py` - CLI and LatentMAS integration helpers.
- `examples/latentmas_stear_hook.py` - upstream LatentMAS patch points.
- `patches/upstream_latentmas_stear.patch` - minimal diff for the official
  LatentMAS `run.py` and `methods/latent_mas.py` files.
- `tests/test_stear_core.py` - focused NumPy unit tests for the controller.

## Minimal upstream integration

In upstream `LatentMAS/run.py`, add STEAR flags after existing parser options:

```python
from latentmas_stear import add_stear_arguments

add_stear_arguments(parser)
```

In upstream `methods/latent_mas.py`, initialize the controller:

```python
from latentmas_stear import build_controller_from_args, apply_stear_to_latent_memory

self.stear_controller = build_controller_from_args(args)
```

In `LatentMASMethod.run_batch_vllm`, after concatenating latent embeddings and
before inserting them into the judger prompt:

```python
past_embedding = torch.cat(embedding_record, dim=1).to(self.vllm_device)
intervention = apply_stear_to_latent_memory(
    past_embedding,
    controller=self.stear_controller,
)
past_embedding = intervention.positive_memory
self.last_stear_decision = intervention.decision
```

If the caller can afford a second decode, `intervention.negative_memory` can be
inserted into a counterfactual prompt and the two output distributions can be
combined with:

```python
calibrated_logits = self.stear_controller.contrastive_logits(
    positive_logits,
    negative_logits,
)
```

## Suggested experiment flag

```bash
python run.py \
  --method latent_mas \
  --model_name Qwen/Qwen3-14B \
  --task gsm8k \
  --prompt sequential \
  --latent_steps 10 \
  --stear \
  --stear_trigger_threshold 0.65 \
  --stear_evidence_ratio 0.25 \
  --stear_injection_strength 0.15
```

## Validation

Run the local tests with:

```bash
python3 -m unittest discover -s tests
```
