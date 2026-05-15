"""Example patch points for SEAL-like STEAR in upstream LatentMAS."""

from latentmas_stear import add_stear_arguments, apply_stear_to_latent_memory, build_controller_from_args


def patch_run_parser(parser):
    """Call this after upstream LatentMAS registers its argparse flags."""

    return add_stear_arguments(parser)


def patch_latentmas_init(self, args):
    """Add this inside ``LatentMASMethod.__init__``."""

    self.stear_controller = build_controller_from_args(args)
    self.last_stear_decision = None


def patch_before_judger_embedding_insert(self, past_embedding, boundary_indices=None):
    """Steer latent memory before the judger sees prior agents' latent states.

    Upstream location:

    ``past_embedding = torch.cat(embedding_record, dim=1).to(self.vllm_device)``

    The default ``--stear_boundary_strategy last`` applies the SEAL-style
    steering vector to the final latent slot, which is the latent analogue of a
    thought boundary before the next reasoning step.
    """

    intervention = apply_stear_to_latent_memory(
        past_embedding,
        controller=self.stear_controller,
        boundary_indices=boundary_indices,
    )
    self.last_stear_decision = intervention.decision
    return intervention.steered_memory
