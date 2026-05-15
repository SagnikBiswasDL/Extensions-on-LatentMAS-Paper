import tempfile
import unittest
from pathlib import Path

import numpy as np

from latentmas_stear import (
    LatentSTEARController,
    STEARConfig,
    classify_reasoning_trace,
    classify_thought,
    compute_reasoning_steering_vector,
    load_steering_vector,
    save_steering_vector,
    split_reasoning_trace,
)


class SealLikeSTEARTest(unittest.TestCase):
    def test_classifies_execution_reflection_and_transition_thoughts(self):
        self.assertEqual(classify_thought("Compute x = 2 + 3."), "execution")
        self.assertEqual(classify_thought("Wait, let me double-check the arithmetic."), "reflection")
        self.assertEqual(classify_thought("Alternatively, solve it with substitution."), "transition")

    def test_splits_and_classifies_reasoning_trace(self):
        trace = "Compute the sum directly.\n\nWait, verify the carry.\n\nAnother approach is to factor it."

        classified = classify_reasoning_trace(trace)

        self.assertEqual([thought for thought, _ in classified], split_reasoning_trace(trace))
        self.assertEqual([label for _, label in classified], ["execution", "reflection", "transition"])

    def test_computes_execution_minus_reflection_transition_vector(self):
        reps = np.array(
            [
                [2.0, 0.0],
                [4.0, 0.0],
                [0.0, 2.0],
                [0.0, 4.0],
            ]
        )
        labels = ["execution", "execution", "reflection", "transition"]

        steering = compute_reasoning_steering_vector(reps, labels, normalize=False)

        np.testing.assert_allclose(steering.vector, np.array([3.0, -3.0]))
        self.assertEqual(steering.execution_count, 2)
        self.assertEqual(steering.reflection_transition_count, 2)

    def test_normalized_steering_vector_has_unit_norm(self):
        reps = np.array([[2.0, 0.0], [0.0, 2.0]])
        labels = ["execution", "reflection"]

        steering = compute_reasoning_steering_vector(reps, labels, normalize=True)

        self.assertAlmostEqual(float(np.linalg.norm(steering.vector)), 1.0)

    def test_apply_to_memory_steers_last_boundary_by_default(self):
        controller = LatentSTEARController(
            STEARConfig(enabled=True, alpha=0.5, boundary_strategy="last", normalize_vector=False),
            steering_vector=np.array([2.0, -2.0]),
        )
        memory = np.zeros((1, 3, 2), dtype=float)

        intervention = controller.apply_to_memory(memory)

        self.assertTrue(intervention.decision.applied)
        np.testing.assert_allclose(intervention.steered_memory[0, 0], np.array([0.0, 0.0]))
        np.testing.assert_allclose(intervention.steered_memory[0, 1], np.array([0.0, 0.0]))
        np.testing.assert_allclose(intervention.steered_memory[0, 2], np.array([1.0, -1.0]))

    def test_apply_to_memory_supports_explicit_boundary_indices(self):
        controller = LatentSTEARController(
            STEARConfig(enabled=True, alpha=1.0, boundary_strategy="indices", normalize_vector=False),
            steering_vector=np.array([1.0, 1.0]),
        )
        memory = np.zeros((2, 4, 2), dtype=float)
        boundary_indices = np.array([[0, 2], [1, 3]])

        intervention = controller.apply_to_memory(memory, boundary_indices=boundary_indices)

        np.testing.assert_allclose(intervention.steered_memory[0, 0], np.array([1.0, 1.0]))
        np.testing.assert_allclose(intervention.steered_memory[0, 1], np.array([0.0, 0.0]))
        np.testing.assert_allclose(intervention.steered_memory[0, 2], np.array([1.0, 1.0]))
        np.testing.assert_allclose(intervention.steered_memory[1, 1], np.array([1.0, 1.0]))
        np.testing.assert_allclose(intervention.steered_memory[1, 3], np.array([1.0, 1.0]))

    def test_disabled_or_missing_vector_leaves_memory_unchanged(self):
        controller = LatentSTEARController(STEARConfig(enabled=False))
        memory = np.ones((1, 2, 3), dtype=float)

        intervention = controller.apply_to_memory(memory)

        self.assertFalse(intervention.decision.applied)
        self.assertIs(intervention.steered_memory, memory)

    def test_save_and_load_steering_vector(self):
        reps = np.array([[1.0, 0.0], [0.0, 1.0]])
        steering = compute_reasoning_steering_vector(reps, ["execution", "transition"], normalize=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stear_vector.npz"
            save_steering_vector(path, steering)
            loaded = load_steering_vector(path)

        np.testing.assert_allclose(loaded, steering.vector)


if __name__ == "__main__":
    unittest.main()
