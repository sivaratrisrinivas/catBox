import os
import json
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from catbox import validate_sd_turbo_runner


class ValidateSdTurboRunnerTests(TestCase):
    def test_validation_command_loads_hf_token_before_model_runner(self):
        with TemporaryDirectory() as project_dir:
            env_file = Path(project_dir) / ".env"
            env_file.write_text("HF_TOKEN=hf_test_token\n", encoding="utf-8")
            observed_token = []

            class TokenAwareRunner:
                def __init__(self, runtime_dir):
                    observed_token.append(os.environ.get("HF_TOKEN"))

                def is_ready(self):
                    return True

                def generate(self, outcome, seed, config=None):
                    return {
                        "image_ref": "/tmp/generated.png",
                        "generation_seconds": 0.1,
                        "metadata": {},
                    }

            argv = [
                "validate_sd_turbo_runner",
                "--outcome",
                "living",
                "--env-file",
                str(env_file),
            ]

            with patch.dict(os.environ, {}, clear=True):
                with patch("sys.argv", argv):
                    with patch(
                        "catbox.validate_sd_turbo_runner.SdTurboImageToImageModelRunner",
                        TokenAwareRunner,
                    ):
                        with patch("sys.stdout", StringIO()):
                            validate_sd_turbo_runner.main()

            self.assertEqual(observed_token, ["hf_test_token"])

    def test_validation_command_passes_single_config_overrides_to_dev_controls(self):
        with TemporaryDirectory() as project_dir:
            env_file = Path(project_dir) / ".env"
            env_file.write_text("", encoding="utf-8")
            observed_generations = []

            class RecordingRunner:
                def __init__(self, runtime_dir):
                    self.runtime_dir = runtime_dir

                def is_ready(self):
                    return True

                def generate(self, outcome, seed, config=None):
                    observed_generations.append(
                        {"outcome": outcome, "seed": seed, "config": config}
                    )
                    return {
                        "image_ref": "/tmp/generated.png",
                        "generation_seconds": 0.1,
                        "metadata": {},
                    }

            argv = [
                "validate_sd_turbo_runner",
                "--outcome",
                "dead",
                "--seed",
                "90210",
                "--steps",
                "6",
                "--strength",
                "0.7",
                "--width",
                "512",
                "--height",
                "512",
                "--env-file",
                str(env_file),
            ]
            stdout = StringIO()

            with patch("sys.argv", argv):
                with patch(
                    "catbox.validate_sd_turbo_runner.SdTurboImageToImageModelRunner",
                    RecordingRunner,
                ):
                    with patch("sys.stdout", stdout):
                        validate_sd_turbo_runner.main()

            payload = json.loads(stdout.getvalue())
            self.assertEqual(
                observed_generations,
                [
                    {
                        "outcome": "dead",
                        "seed": 90210,
                        "config": {
                            "steps": 6,
                            "strength": 0.7,
                            "width": 512,
                            "height": 512,
                        },
                    }
                ],
            )
            self.assertEqual(
                payload[1]["metadata"]["configOverrides"],
                {
                    "steps": 6,
                    "strength": 0.7,
                    "width": 512,
                    "height": 512,
                },
            )

    def test_validation_command_runs_matrix_with_outcome_specific_strengths(self):
        with TemporaryDirectory() as project_dir:
            env_file = Path(project_dir) / ".env"
            env_file.write_text("", encoding="utf-8")
            observed_generations = []

            class RecordingRunner:
                def __init__(self, runtime_dir):
                    self.runtime_dir = runtime_dir

                def is_ready(self):
                    return True

                def generate(self, outcome, seed, config=None):
                    observed_generations.append(
                        {"outcome": outcome, "seed": seed, "config": config}
                    )
                    return {
                        "image_ref": f"/tmp/{outcome}_{len(observed_generations)}.png",
                        "generation_seconds": 0.1,
                        "metadata": {},
                    }

            argv = [
                "validate_sd_turbo_runner",
                "--matrix",
                "--seed",
                "41100",
                "--matrix-steps",
                "4,6",
                "--matrix-sizes",
                "512",
                "--living-strengths",
                "0.75",
                "--dead-strengths",
                "0.6,0.7",
                "--env-file",
                str(env_file),
            ]
            stdout = StringIO()

            with patch("sys.argv", argv):
                with patch(
                    "catbox.validate_sd_turbo_runner.SdTurboImageToImageModelRunner",
                    RecordingRunner,
                ):
                    with patch("sys.stdout", stdout):
                        validate_sd_turbo_runner.main()

            payload = json.loads(stdout.getvalue())

            self.assertEqual(
                observed_generations,
                [
                    {
                        "outcome": "living",
                        "seed": 41100,
                        "config": {
                            "steps": 4,
                            "strength": 0.75,
                            "width": 512,
                            "height": 512,
                        },
                    },
                    {
                        "outcome": "living",
                        "seed": 41100,
                        "config": {
                            "steps": 6,
                            "strength": 0.75,
                            "width": 512,
                            "height": 512,
                        },
                    },
                    {
                        "outcome": "dead",
                        "seed": 41100,
                        "config": {
                            "steps": 4,
                            "strength": 0.6,
                            "width": 512,
                            "height": 512,
                        },
                    },
                    {
                        "outcome": "dead",
                        "seed": 41100,
                        "config": {
                            "steps": 4,
                            "strength": 0.7,
                            "width": 512,
                            "height": 512,
                        },
                    },
                    {
                        "outcome": "dead",
                        "seed": 41100,
                        "config": {
                            "steps": 6,
                            "strength": 0.6,
                            "width": 512,
                            "height": 512,
                        },
                    },
                    {
                        "outcome": "dead",
                        "seed": 41100,
                        "config": {
                            "steps": 6,
                            "strength": 0.7,
                            "width": 512,
                            "height": 512,
                        },
                    },
                ],
            )
            self.assertEqual(payload[0]["readiness"]["status"], "ready")
            self.assertEqual(len(payload[1:]), 6)
            self.assertEqual(
                payload[1]["metadata"]["configOverrides"],
                {
                    "steps": 4,
                    "strength": 0.75,
                    "width": 512,
                    "height": 512,
                },
            )
