from pathlib import Path
from unittest import TestCase


ROOT = Path(__file__).resolve().parent.parent


class ManualValidationDocumentationTests(TestCase):
    def test_manual_gpu_validation_documents_first_observation_flow(self):
        guide_path = ROOT / "docs" / "manual-gpu-validation.md"
        self.assertTrue(guide_path.exists())

        guide = guide_path.read_text(encoding="utf-8")

        self.assertIn("python -m catbox.browser_ui", guide)
        self.assertIn("http://127.0.0.1:8765", guide)
        self.assertIn("POST /api/observe", guide)
        self.assertIn("python -m catbox.validate_sd_turbo_runner --outcome all --seed 41100", guide)
        self.assertIn("python -m catbox.validate_sd_turbo_runner --matrix", guide)
        self.assertIn("Primary Runtime Target", guide)
        self.assertIn("under 23 seconds", guide)
        self.assertIn(".runtime/generated-outcomes", guide)
        self.assertIn("no gallery/history/save/share behavior", guide)
        self.assertIn("Retry", guide)
        self.assertIn("Reset", guide)
        self.assertIn("HF_TOKEN", guide)
