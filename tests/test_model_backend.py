from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from catbox.model_backend import CatboxModelBackend, FakeModelRunner


class FailingModelRunner(FakeModelRunner):
    def generate(self, outcome, seed, config=None):
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

    def test_dev_controls_can_force_living_outcome(self):
        with TemporaryDirectory() as output_dir:
            runner = FakeModelRunner(output_dir=output_dir)
            backend = CatboxModelBackend(
                model_runner=runner,
                seed_source=lambda: 41100,
                outcome_source=lambda: "absent",
            )

            response = backend.observe_with_dev_controls({"outcome": "living"})

            self.assertEqual(response["status"], "generated")
            self.assertEqual(response["outcome"], "living")
            self.assertEqual(runner.generations, [{"outcome": "living", "seed": 41100}])

    def test_dev_controls_can_force_absent_outcome(self):
        with TemporaryDirectory() as output_dir:
            runner = FakeModelRunner(output_dir=output_dir)
            backend = CatboxModelBackend(
                model_runner=runner,
                seed_source=lambda: 41100,
                outcome_source=lambda: "living",
            )

            response = backend.observe_with_dev_controls({"outcome": "absent"})

            self.assertEqual(response["status"], "generated")
            self.assertEqual(response["outcome"], "absent")
            self.assertEqual(runner.generations, [{"outcome": "absent", "seed": 41100}])

    def test_dev_controls_can_override_seed_for_reproducible_observation(self):
        with TemporaryDirectory() as output_dir:
            runner = FakeModelRunner(output_dir=output_dir)
            backend = CatboxModelBackend(
                model_runner=runner,
                seed_source=lambda: 41100,
                outcome_source=lambda: "living",
            )

            response = backend.observe_with_dev_controls({"seed": 90210})

            self.assertEqual(response["status"], "generated")
            self.assertEqual(response["metadata"]["seed"], 90210)
            self.assertEqual(runner.generations, [{"outcome": "living", "seed": 90210}])

    def test_dev_controls_pass_config_overrides_to_generation_metadata_and_runner(self):
        with TemporaryDirectory() as output_dir:
            runner = FakeModelRunner(output_dir=output_dir)
            backend = CatboxModelBackend(
                model_runner=runner,
                seed_source=lambda: 41100,
                outcome_source=lambda: "absent",
            )

            response = backend.observe_with_dev_controls(
                {"config": {"steps": 6, "strength": 0.55}}
            )

            self.assertEqual(response["status"], "generated")
            self.assertEqual(
                response["metadata"]["configOverrides"],
                {"steps": 6, "strength": 0.55},
            )
            self.assertEqual(
                runner.generations,
                [
                    {
                        "outcome": "absent",
                        "seed": 41100,
                        "config": {"steps": 6, "strength": 0.55},
                    }
                ],
            )

    def test_invalid_dev_controls_outcome_fails_clearly_without_generation(self):
        with TemporaryDirectory() as output_dir:
            runner = FakeModelRunner(output_dir=output_dir)
            backend = CatboxModelBackend(
                model_runner=runner,
                seed_source=lambda: 41100,
            )

            response = backend.observe_with_dev_controls({"outcome": "ghost"})

            self.assertEqual(response["status"], "generation_failed")
            self.assertEqual(response["error"]["type"], "InvalidDevControlsOverride")
            self.assertEqual(response["error"]["field"], "outcome")
            self.assertIn("ghost", response["error"]["message"])
            self.assertNotIn("imageRef", response)
            self.assertEqual(runner.generations, [])

    def test_invalid_dev_controls_seed_fails_clearly_without_generation(self):
        with TemporaryDirectory() as output_dir:
            runner = FakeModelRunner(output_dir=output_dir)
            backend = CatboxModelBackend(
                model_runner=runner,
                outcome_source=lambda: "living",
            )

            response = backend.observe_with_dev_controls({"seed": "not-a-seed"})

            self.assertEqual(response["status"], "generation_failed")
            self.assertEqual(response["error"]["type"], "InvalidDevControlsOverride")
            self.assertEqual(response["error"]["field"], "seed")
            self.assertIn("integer", response["error"]["message"])
            self.assertNotIn("imageRef", response)
            self.assertEqual(runner.generations, [])

    def test_invalid_dev_controls_config_fails_clearly_without_generation(self):
        with TemporaryDirectory() as output_dir:
            runner = FakeModelRunner(output_dir=output_dir)
            backend = CatboxModelBackend(
                model_runner=runner,
                outcome_source=lambda: "living",
            )

            response = backend.observe_with_dev_controls({"config": ["steps", 6]})

            self.assertEqual(response["status"], "generation_failed")
            self.assertEqual(response["error"]["type"], "InvalidDevControlsOverride")
            self.assertEqual(response["error"]["field"], "config")
            self.assertIn("object", response["error"]["message"])
            self.assertNotIn("imageRef", response)
            self.assertEqual(runner.generations, [])
