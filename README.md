# Extensions on LatentMAS: STEAR for Latent Reasoning

This repository contains a lightweight implementation of **STEAR as a
SEAL-like steering method** for LatentMAS. The goal is not tool calling; it is
to test whether representation-level reasoning calibration improves latent
multi-agent reasoning.

## What is implemented

Here, "STEAR" means the same family of method as SEAL (*Steerable Reasoning
Calibration*):

1. split reasoning traces into thought blocks;
2. classify thoughts as **execution**, **reflection**, or **transition**;
3. extract a steering vector in latent space:

   `S = mean(H_execution) - mean(H_reflection_or_transition)`

4. during inference, calibrate hidden states with:

   `H_steered = H + alpha * S`

LatentMAS communicates through hidden-state working memory rather than visible
chain-of-thought tokens. This extension adapts SEAL-like STEAR to that setting:

- **Thought labeling utilities**: keyword-based classification matching SEAL's
  execution/reflection/transition categories.
- **Steering-vector extraction**: computes the execution-minus-nonexecution
  direction from thought-boundary hidden states.
- **Latent memory intervention**: applies `alpha * S` to LatentMAS working
  memory slots at reasoning boundaries.
- **Boundary strategies**: steer only the final latent slot (`last`), every slot
  (`all`), or explicit slot indices (`indices`).

The code is backend-light: it is tested with NumPy and works with PyTorch
tensors in a LatentMAS runtime without importing PyTorch unless tensor inputs
come from PyTorch.

## Files

- `latentmas_stear/core.py` - thought classification, steering-vector
  extraction, and latent intervention.
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
past_embedding = intervention.steered_memory
self.last_stear_decision = intervention.decision
```

## Steering vector extraction

Collect hidden states at thought boundaries from an offline calibration set,
label each thought, then compute and save the vector:

```python
from latentmas_stear import compute_reasoning_steering_vector, save_steering_vector

steering = compute_reasoning_steering_vector(
    thought_boundary_hidden_states,  # shape: [num_thoughts, hidden]
    thought_labels,                  # execution/reflection/transition
)
save_steering_vector("stear_vector.npz", steering)
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
  --stear_vector_path stear_vector.npz \
  --stear_alpha 1.0 \
  --stear_boundary_strategy last
```

## Validation

Run the local tests with:

```bash
python3 -m unittest discover -s tests
```
