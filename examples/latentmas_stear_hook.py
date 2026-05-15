"""Example patch points for adding STEAR to upstream LatentMAS.

This file is intentionally not an executable replacement for LatentMAS. It
shows the minimal integration points in the upstream project:

1. add CLI arguments in ``run.py``;
2. create a controller in ``LatentMASMethod.__init__``;
3. apply it to the latent ``past_embedding`` before judger decoding.
"""

from latentmas_stear import add_stear_arguments, apply_stear_to_latent_memory, build_controller_from_args


def patch_run_parser(parser):
    """Call this after LatentMAS registers its existing argparse flags."""

    return add_stear_arguments(parser)


def patch_latentmas_init(self, args):
    """Add this inside ``LatentMASMethod.__init__``."""

    self.stear_controller = build_controller_from_args(args)


def patch_before_judger_embedding_insert(self, past_embedding, judger_query=None):
    """Add this in ``LatentMASMethod.run_batch_vllm`` before prompt insertion.

    Upstream location:

    ``past_embedding = torch.cat(embedding_record, dim=1).to(self.vllm_device)``

    Immediately after that line, call this helper and then continue using the
    returned positive memory as ``past_embedding``.
    """

    intervention = apply_stear_to_latent_memory(
        past_embedding,
        controller=self.stear_controller,
        query=judger_query,
    )
    past_embedding = intervention.positive_memory

    # Optional: store metadata in the run trace for ablations and diagnostics.
    self.last_stear_decision = intervention.decision
    return past_embedding
