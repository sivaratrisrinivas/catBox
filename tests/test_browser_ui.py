import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from urllib.parse import quote

from catbox.browser_ui import BrowserUiApp


class RecordingBackend:
    def __init__(self, image_ref: str) -> None:
        self.image_ref = image_ref
        self.observations = 0

    def observe(self):
        self.observations += 1
        return {
            "status": "generated",
            "outcome": "living",
            "imageRef": self.image_ref,
            "metadata": {
                "seed": 41100,
                "ephemeral": True,
            },
            "revealNote": "A local diffusion model generated this outcome.",
        }


class BrowserUiServerTests(TestCase):
    def test_browser_page_renders_observation_flow_without_outcome_choice(self):
        backend = RecordingBackend("/tmp/generated.png")
        app = BrowserUiApp(backend=backend)

        response = app.handle("GET", "/")
        html = response.body.decode("utf-8")

        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers["Content-Type"], "text/html; charset=utf-8")
        self.assertIn("data-state=\"sealed\"", html)
        self.assertIn("id=\"observe-button\"", html)
        self.assertIn("Observe", html)
        self.assertIn("data-state=\"waiting\"", html)
        self.assertIn("Observation noise", html)
        self.assertIn("id=\"generated-outcome\"", html)
        self.assertIn("id=\"reveal-note\"", html)
        self.assertIn("id=\"reset-button\"", html)
        self.assertNotIn("name=\"outcome\"", html)

    def test_normal_observation_returns_backend_contract_and_serves_generated_file(self):
        with TemporaryDirectory() as runtime_dir:
            image_path = Path(runtime_dir) / "generated.png"
            image_path.write_bytes(b"generated image")
            backend = RecordingBackend(str(image_path))
            app = BrowserUiApp(backend=backend)

            observe_response = app.handle("POST", "/api/observe")
            payload = json.loads(observe_response.body.decode("utf-8"))

            self.assertEqual(observe_response.status, 200)
            self.assertEqual(observe_response.headers["Content-Type"], "application/json")
            self.assertEqual(backend.observations, 1)
            self.assertEqual(payload["status"], "generated")
            self.assertEqual(payload["outcome"], "living")
            self.assertEqual(payload["imageRef"], str(image_path))
            self.assertEqual(payload["metadata"]["seed"], 41100)
            self.assertEqual(payload["revealNote"], "A local diffusion model generated this outcome.")

            image_response = app.handle(
                "GET",
                f"/api/generated-outcome?imageRef={quote(payload['imageRef'])}",
            )
            self.assertEqual(image_response.status, 200)
            self.assertEqual(image_response.body, b"generated image")
            self.assertEqual(image_response.headers["Content-Type"], "image/png")
