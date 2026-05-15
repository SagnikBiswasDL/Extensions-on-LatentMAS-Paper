import unittest

import numpy as np

from latentmas_stear import LatentSTEARController, STEARConfig


class LatentSTEARControllerTest(unittest.TestCase):
    def test_uncertainty_trigger_uses_entropy_and_margin(self):
        controller = LatentSTEARController(STEARConfig(enabled=True, trigger_threshold=0.8, margin_threshold=0.1))
        confident_logits = np.array([[8.0, 0.0, -1.0]])
        uncertain_logits = np.array([[0.1, 0.0, -0.1]])

        confident_entropy, confident_margin = controller.uncertainty_from_logits(confident_logits)
        uncertain_entropy, uncertain_margin = controller.uncertainty_from_logits(uncertain_logits)

        self.assertLess(confident_entropy[0], 0.2)
        self.assertGreater(confident_margin[0], 0.9)
        self.assertGreater(uncertain_entropy[0], 0.95)
        self.assertLess(uncertain_margin[0], 0.1)
        self.assertFalse(controller.trigger_mask(logits=confident_logits)[0])
        self.assertTrue(controller.trigger_mask(logits=uncertain_logits)[0])

    def test_select_key_evidence_prefers_query_aligned_memory(self):
        controller = LatentSTEARController(STEARConfig(enabled=True, evidence_ratio=0.5, max_evidence_tokens=2))
        memory = np.array(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.9, 0.1, 0.0],
                    [-1.0, 0.0, 0.0],
                ]
            ]
        )
        query = np.array([[1.0, 0.0, 0.0]])

        indices, scores = controller.select_key_evidence(memory, query=query)

        self.assertEqual(indices.shape, (1, 2))
        self.assertEqual(indices[0, 0], 0)
        self.assertEqual(indices[0, 1], 2)
        self.assertGreater(scores[0, 0], scores[0, 1])

    def test_reinject_moves_target_toward_selected_evidence(self):
        controller = LatentSTEARController(STEARConfig(enabled=True, injection_strength=0.5))
        target = np.array([[0.0, 1.0]])
        evidence = np.array([[[1.0, 0.0], [1.0, 0.0]]])

        updated = controller.reinject(target, evidence)

        self.assertEqual(updated.shape, target.shape)
        self.assertGreater(updated[0, 0], target[0, 0])
        self.assertEqual(updated[0, 1], target[0, 1])

    def test_counterfactual_reverses_only_selected_memory(self):
        controller = LatentSTEARController(STEARConfig(enabled=True, counterfactual_mode="reverse"))
        memory = np.arange(12, dtype=float).reshape(1, 4, 3)
        indices = np.array([[1, 3]])

        counterfactual = controller.build_counterfactual_memory(memory, indices)

        np.testing.assert_array_equal(counterfactual[0, 0], memory[0, 0])
        np.testing.assert_array_equal(counterfactual[0, 2], memory[0, 2])
        np.testing.assert_array_equal(counterfactual[0, 1], memory[0, 3])
        np.testing.assert_array_equal(counterfactual[0, 3], memory[0, 1])

    def test_contrastive_logits_follow_stear_formula(self):
        controller = LatentSTEARController(STEARConfig(enabled=True, counterfactual_alpha=0.25))
        positive = np.array([[2.0, 1.0]])
        negative = np.array([[0.0, 4.0]])

        calibrated = controller.contrastive_logits(positive, negative)

        np.testing.assert_allclose(calibrated, np.array([[2.5, 0.25]]))

    def test_intervention_returns_positive_and_negative_memory_when_triggered(self):
        controller = LatentSTEARController(
            STEARConfig(enabled=True, trigger_threshold=0.8, margin_threshold=0.1, injection_strength=0.5)
        )
        memory = np.array(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                ]
            ]
        )
        uncertain_logits = np.array([[0.0, 0.0, 0.0]])

        intervention = controller.intervene_latent_memory(memory, logits=uncertain_logits)

        self.assertTrue(intervention.decision.triggered[0])
        self.assertIsNotNone(intervention.negative_memory)
        self.assertEqual(intervention.positive_memory.shape, memory.shape)
        self.assertGreater(intervention.positive_memory[0, -1, 0], memory[0, -1, 0])


if __name__ == "__main__":
    unittest.main()
