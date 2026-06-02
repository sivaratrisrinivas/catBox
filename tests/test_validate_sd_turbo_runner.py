import os
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
