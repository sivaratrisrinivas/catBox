from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from catbox.model_backend import CatboxModelBackend, FakeModelRunner
from catbox.sd_turbo_runner import SdTurboImageToImageModelRunner


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
            self.assertIn(response["outcome"], {"living", "dead"})
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
                outcome_source=lambda: "dead",
            )

            response = backend.observe()

            self.assertEqual(response["status"], "generated")
            self.assertEqual(runner.generations, [{"outcome": "dead", "seed": 41100}])

    def test_dev_controls_can_force_living_outcome(self):
        with TemporaryDirectory() as output_dir:
            runner = FakeModelRunner(output_dir=output_dir)
            backend = CatboxModelBackend(
                model_runner=runner,
                seed_source=lambda: 41100,
                outcome_source=lambda: "dead",
            )

            response = backend.observe_with_dev_controls({"outcome": "living"})

            self.assertEqual(response["status"], "generated")
            self.assertEqual(response["outcome"], "living")
            self.assertEqual(runner.generations, [{"outcome": "living", "seed": 41100}])

    def test_dev_controls_can_force_dead_outcome(self):
        with TemporaryDirectory() as output_dir:
            runner = FakeModelRunner(output_dir=output_dir)
            backend = CatboxModelBackend(
                model_runner=runner,
                seed_source=lambda: 41100,
                outcome_source=lambda: "living",
            )

            response = backend.observe_with_dev_controls({"outcome": "dead"})

            self.assertEqual(response["status"], "generated")
            self.assertEqual(response["outcome"], "dead")
            self.assertEqual(runner.generations, [{"outcome": "dead", "seed": 41100}])

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
                outcome_source=lambda: "dead",
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
                        "outcome": "dead",
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


class StubGeneratedImage:
    def __init__(self) -> None:
        self.saved_path = None

    def save(self, path):
        self.saved_path = path
        Path(path).write_bytes(b"generated image")


class StubPipeline:
    def __init__(self) -> None:
        self.calls = []
        self.generated_image = StubGeneratedImage()

    def to(self, device):
        self.device = device
        return self

    def enable_attention_slicing(self):
        self.attention_slicing_enabled = True

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return type("PipelineResult", (), {"images": [self.generated_image]})()


class StubTraceLatents:
    def detach(self):
        return self

    def __truediv__(self, value):
        return self


class StubTraceVae:
    config = type("VaeConfig", (), {"scaling_factor": 1.0})()

    def decode(self, latents):
        return type("Decoded", (), {"sample": "decoded image"})()


class StubTraceImageProcessor:
    def postprocess(self, decoded, output_type):
        return [StubGeneratedImage()]


class StubTracePipeline(StubPipeline):
    def __init__(self) -> None:
        super().__init__()
        self.vae = StubTraceVae()
        self.image_processor = StubTraceImageProcessor()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        callback = kwargs.get("callback_on_step_end")
        if callback is not None:
            callback(self, 0, "timestep", {"latents": StubTraceLatents()})
        return type("PipelineResult", (), {"images": [self.generated_image]})()


class StubTorch:
    float16 = "float16"
    float32 = "float32"

    class cuda:
        @staticmethod
        def is_available():
            return False

    class Generator:
        def __init__(self, device):
            self.device = device
            self.seed = None

        def manual_seed(self, seed):
            self.seed = seed
            return self

    class no_grad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            return False


class SdTurboRunnerTests(TestCase):
    def test_real_runner_generates_selected_outcome_file_with_prompt_and_metadata(self):
        with TemporaryDirectory() as runtime_dir:
            pipeline = StubPipeline()
            loaded = []
            box_image_loads = []

            def load_pipeline(model_id, **kwargs):
                loaded.append({"model_id": model_id, "kwargs": kwargs})
                return pipeline

            def load_box_image(path, config):
                box_image_loads.append({"width": config.width, "height": config.height})
                return "box image"

            runner = SdTurboImageToImageModelRunner(
                runtime_dir=runtime_dir,
                pipeline_loader=load_pipeline,
                box_image_loader=load_box_image,
                torch_module=StubTorch,
                now=lambda: "2026-06-02T12:00:00Z",
                timer=iter([10.0, 10.25]).__next__,
            )

            generated = runner.generate("living", seed=41100)

            self.assertTrue(runner.is_ready())
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["model_id"], "stabilityai/sd-turbo")
            self.assertEqual(generated["generation_seconds"], 0.25)
            self.assertTrue(Path(generated["image_ref"]).exists())
            self.assertIn("living cat", pipeline.calls[0]["prompt"])
            self.assertEqual(pipeline.calls[0]["num_inference_steps"], 4)
            self.assertEqual(pipeline.calls[0]["strength"], 0.8)
            self.assertEqual(box_image_loads, [{"width": 384, "height": 384}])
            self.assertNotIn("negative_prompt", pipeline.calls[0])
            self.assertEqual(generated["metadata"]["runner"], "sd_turbo_img2img")
            self.assertEqual(generated["metadata"]["device"], "cpu")

    def test_backend_response_includes_real_runner_metadata(self):
        with TemporaryDirectory() as runtime_dir:
            pipeline = StubPipeline()
            box_image_loads = []

            runner = SdTurboImageToImageModelRunner(
                runtime_dir=runtime_dir,
                pipeline_loader=lambda model_id, **kwargs: pipeline,
                box_image_loader=lambda path, config: box_image_loads.append(
                    {"width": config.width, "height": config.height}
                )
                or "box image",
                torch_module=StubTorch,
                now=lambda: "2026-06-02T12:00:00Z",
                timer=iter([20.0, 20.5]).__next__,
            )
            backend = CatboxModelBackend(
                model_runner=runner,
                seed_source=lambda: 41100,
                outcome_source=lambda: "dead",
                clock=lambda: 123.0,
            )

            response = backend.observe()

            self.assertEqual(response["status"], "generated")
            self.assertEqual(response["outcome"], "dead")
            self.assertTrue(Path(response["imageRef"]).exists())
            self.assertEqual(response["metadata"]["seed"], 41100)
            self.assertEqual(response["metadata"]["generationSeconds"], 0.5)
            self.assertEqual(response["metadata"]["runner"], "sd_turbo_img2img")
            self.assertEqual(response["metadata"]["device"], "cpu")
            self.assertIn("deceased cat", pipeline.calls[0]["prompt"])
            self.assertIn("no blood", pipeline.calls[0]["prompt"])
            self.assertIn("no gore", pipeline.calls[0]["prompt"])
            self.assertEqual(pipeline.calls[0]["num_inference_steps"], 2)
            self.assertEqual(pipeline.calls[0]["strength"], 0.55)
            self.assertEqual(box_image_loads, [{"width": 512, "height": 512}])

    def test_real_runner_captures_selected_branch_trace_frames(self):
        with TemporaryDirectory() as runtime_dir:
            pipeline = StubTracePipeline()
            observed_trace_refs = []

            runner = SdTurboImageToImageModelRunner(
                runtime_dir=runtime_dir,
                pipeline_loader=lambda model_id, **kwargs: pipeline,
                box_image_loader=lambda path, config: "box image",
                torch_module=StubTorch,
                now=lambda: "2026-06-02T12:00:00Z",
                timer=iter([22.0, 22.4]).__next__,
            )

            generated = runner.generate(
                "living",
                seed=41100,
                trace_callback=observed_trace_refs.append,
            )

            self.assertEqual(len(generated["trace_refs"]), 1)
            self.assertEqual(observed_trace_refs, generated["trace_refs"])
            self.assertTrue(Path(generated["trace_refs"][0]).exists())
            self.assertIn("callback_on_step_end", pipeline.calls[0])
            self.assertEqual(
                pipeline.calls[0]["callback_on_step_end_tensor_inputs"],
                ["latents"],
            )

    def test_backend_reports_starting_and_failure_when_real_runner_preload_fails(self):
        with TemporaryDirectory() as runtime_dir:
            def fail_to_load(model_id, **kwargs):
                raise RuntimeError("model weights unavailable")

            runner = SdTurboImageToImageModelRunner(
                runtime_dir=runtime_dir,
                pipeline_loader=fail_to_load,
                box_image_loader=lambda path, config: "box image",
                torch_module=StubTorch,
            )
            backend = CatboxModelBackend(
                model_runner=runner,
                seed_source=lambda: 41100,
                outcome_source=lambda: "living",
            )

            readiness = backend.readiness()
            response = backend.observe()

            self.assertEqual(readiness["status"], "starting")
            self.assertEqual(response["status"], "generation_failed")
            self.assertEqual(response["error"]["type"], "RuntimeError")
            self.assertIn("model weights unavailable", response["error"]["message"])
            self.assertNotIn("imageRef", response)

    def test_real_runner_applies_generation_config_overrides(self):
        with TemporaryDirectory() as runtime_dir:
            pipeline = StubPipeline()
            runner = SdTurboImageToImageModelRunner(
                runtime_dir=runtime_dir,
                pipeline_loader=lambda model_id, **kwargs: pipeline,
                box_image_loader=lambda path, config: "box image",
                torch_module=StubTorch,
                now=lambda: "2026-06-02T12:00:00Z",
                timer=iter([30.0, 30.75]).__next__,
            )

            generated = runner.generate(
                "dead",
                seed=41100,
                config={"steps": 6, "strength": 0.42, "ignored": "not forwarded"},
            )

            self.assertEqual(pipeline.calls[0]["num_inference_steps"], 6)
            self.assertEqual(pipeline.calls[0]["strength"], 0.42)
            self.assertEqual(generated["metadata"]["config"]["steps"], 6)
            self.assertEqual(generated["metadata"]["config"]["strength"], 0.42)
            self.assertNotIn("ignored", generated["metadata"]["config"])
