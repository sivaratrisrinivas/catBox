from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from catbox.model_backend import CatboxModelBackend, FakeModelRunner


class FailingModelRunner(FakeModelRunner):
    def generate(self, outcome, seed):
        raise RuntimeError("CUDA ran out of memory")


class ModelBackendTests(TestCase):
    def test_readiness_reports_starting_until_runner_is_ready(self):
        with TemporaryDirectory() as output_dir:
            backend = CatboxModelBackend(
                model_runner=FakeModelRunner(output_dir=output_dir, ready=False)
            )

            response = backend.readiness()

            self.assertEqual(response["status"], "starting")
            self.assertEqual(response["modelBackend"], "starting")

    def test_observation_returns_generated_outcome_contract(self):
        with TemporaryDirectory() as output_dir:
            backend = CatboxModelBackend(
                model_runner=FakeModelRunner(output_dir=output_dir),
                seed_source=lambda: 41100,
                clock=lambda: 12.5,
            )

            response = backend.observe()

            self.assertEqual(response["status"], "generated")
            self.assertIn(response["outcome"], {"living", "absent"})
            self.assertTrue(response["imageRef"].endswith(".png"))
            self.assertTrue(Path(response["imageRef"]).exists())
            self.assertEqual(response["metadata"]["seed"], 41100)
            self.assertIs(response["metadata"]["ephemeral"], True)
            self.assertTrue(response["revealNote"])

    def test_generation_failure_is_structured_without_fake_outcome(self):
        with TemporaryDirectory() as output_dir:
            backend = CatboxModelBackend(
                model_runner=FailingModelRunner(output_dir=output_dir),
                seed_source=lambda: 41100,
                outcome_source=lambda: "living",
            )

            response = backend.observe()

            self.assertEqual(response["status"], "generation_failed")
            self.assertEqual(response["error"]["type"], "RuntimeError")
            self.assertIn("CUDA", response["error"]["message"])
            self.assertNotIn("imageRef", response)
            self.assertEqual(response["metadata"]["outcome"], "living")
            self.assertIs(response["metadata"]["ephemeral"], True)

    def test_invalid_outcome_selection_fails_before_generation(self):
        with TemporaryDirectory() as output_dir:
            runner = FakeModelRunner(output_dir=output_dir)
            backend = CatboxModelBackend(
                model_runner=runner,
                seed_source=lambda: 41100,
                outcome_source=lambda: "ghost",
            )

            response = backend.observe()

            self.assertEqual(response["status"], "generation_failed")
            self.assertEqual(response["error"]["type"], "InvalidOutcome")
            self.assertNotIn("imageRef", response)

    def test_observation_invokes_runner_once_for_selected_outcome(self):
        with TemporaryDirectory() as output_dir:
            runner = FakeModelRunner(output_dir=output_dir)
            backend = CatboxModelBackend(
                model_runner=runner,
                seed_source=lambda: 41100,
                outcome_source=lambda: "absent",
            )

            response = backend.observe()

            self.assertEqual(response["status"], "generated")
            self.assertEqual(runner.generations, [{"outcome": "absent", "seed": 41100}])
