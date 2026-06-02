import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from urllib.parse import quote

from catbox.browser_ui import BrowserUiApp, make_default_browser_ui_server


class RecordingBackend:
    def __init__(self, image_ref: str, ready: bool = True) -> None:
        self.image_ref = image_ref
        self.ready = ready
        self.observations = 0

    def readiness(self):
        if self.ready:
            return {"status": "ready", "modelBackend": "ready"}
        return {"status": "starting", "modelBackend": "starting"}

    def observe(self, trace_callback=None):
        self.observations += 1
        if trace_callback is not None:
            trace_callback(self.image_ref)
        return {
            "status": "generated",
            "outcome": "living",
            "imageRef": self.image_ref,
            "traceRefs": [self.image_ref] if trace_callback is not None else [],
            "metadata": {
                "seed": 41100,
                "ephemeral": True,
                "traceFrameCount": 1 if trace_callback is not None else 0,
            },
            "revealNote": "A local diffusion model generated this outcome.",
        }


class FailingBackend:
    def __init__(self) -> None:
        self.observations = 0

    def readiness(self):
        return {"status": "ready", "modelBackend": "ready"}

    def observe(self, trace_callback=None):
        self.observations += 1
        return {
            "status": "generation_failed",
            "error": {
                "type": "RuntimeError",
                "message": "CUDA ran out of memory",
            },
            "metadata": {
                "seed": 41100,
                "outcome": "living",
                "ephemeral": True,
            },
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
        self.assertIn("data-state=\"starting\"", html)
        self.assertIn("Preparing the local diffusion chamber", html)
        self.assertIn("id=\"observe-button\"", html)
        self.assertIn("Observe", html)
        self.assertIn("data-state=\"waiting\"", html)
        self.assertIn("Captured Denoising Trace", html)
        self.assertIn("id=\"trace-frame\"", html)
        self.assertIn("id=\"trace-placeholder\"", html)
        self.assertIn("TRACE_POLL_MS", html)
        self.assertIn("pollTrace", html)
        self.assertIn("/api/trace", html)
        self.assertIn("id=\"progressive-waiting-status\"", html)
        self.assertIn("hidden", html)
        self.assertIn("PROGRESSIVE_WAITING_DELAY_MS", html)
        self.assertIn("showProgressiveWaiting", html)
        self.assertIn("id=\"generated-outcome\"", html)
        self.assertIn("data-loading=\"true\"", html)
        self.assertIn("generatedOutcome.onload", html)
        self.assertIn("generatedOutcome.onerror", html)
        self.assertIn("id=\"reveal-note\"", html)
        self.assertIn("id=\"reset-button\"", html)
        self.assertIn("data-state=\"generation-failure\"", html)
        self.assertIn("id=\"generation-failure-message\"", html)
        self.assertIn("id=\"retry-button\"", html)
        self.assertIn("id=\"failure-reset-button\"", html)
        self.assertIn("retryButton.addEventListener(\"click\", observe)", html)
        self.assertIn("failureResetButton.addEventListener(\"click\", resetToSealed)", html)
        self.assertIn("generationFailureMessage.textContent = \"\"", html)
        self.assertIn("transform: scale(0.97)", html)
        self.assertIn("@media (prefers-reduced-motion: reduce)", html)
        self.assertIn("@media (hover: hover) and (pointer: fine)", html)
        self.assertIn("overflow-y: auto", html)
        self.assertIn("grid-template-rows: auto minmax(0, 1fr) auto", html)
        self.assertIn(".panel {\n      display: none", html)
        self.assertIn(".panel.is-active {\n      display: grid", html)
        self.assertIn("bringActivePanelIntoView(revealedPanel)", html)
        self.assertIn(".brand {\n      position: relative", html)
        self.assertIn(".hud {\n      position: relative", html)
        self.assertNotIn("name=\"outcome\"", html)

    def test_readiness_endpoint_returns_backend_startup_state(self):
        backend = RecordingBackend("/tmp/generated.png", ready=False)
        app = BrowserUiApp(backend=backend)

        response = app.handle("GET", "/api/readiness")
        payload = json.loads(response.body.decode("utf-8"))

        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers["Content-Type"], "application/json")
        self.assertEqual(payload, {"status": "starting", "modelBackend": "starting"})

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
            self.assertEqual(payload["traceRefs"], [str(image_path)])
            self.assertEqual(payload["metadata"]["seed"], 41100)
            self.assertEqual(payload["metadata"]["traceFrameCount"], 1)
            self.assertEqual(payload["revealNote"], "A local diffusion model generated this outcome.")

            trace_response = app.handle("GET", "/api/trace")
            trace_payload = json.loads(trace_response.body.decode("utf-8"))
            self.assertEqual(trace_response.status, 200)
            self.assertEqual(trace_payload, {"traceRefs": [str(image_path)]})

            image_response = app.handle(
                "GET",
                f"/api/generated-outcome?imageRef={quote(payload['imageRef'])}",
            )
            self.assertEqual(image_response.status, 200)
            self.assertEqual(image_response.body, b"generated image")
            self.assertEqual(image_response.headers["Content-Type"], "image/png")

    def test_generation_failure_returns_structured_failure_without_generated_image(self):
        backend = FailingBackend()
        app = BrowserUiApp(backend=backend)

        observe_response = app.handle("POST", "/api/observe")
        payload = json.loads(observe_response.body.decode("utf-8"))

        self.assertEqual(observe_response.status, 200)
        self.assertEqual(observe_response.headers["Content-Type"], "application/json")
        self.assertEqual(backend.observations, 1)
        self.assertEqual(payload["status"], "generation_failed")
        self.assertEqual(payload["error"]["type"], "RuntimeError")
        self.assertIn("CUDA", payload["error"]["message"])
        self.assertNotIn("imageRef", payload)

        image_response = app.handle(
            "GET",
            "/api/generated-outcome?imageRef=/tmp/not-generated.png",
        )
        self.assertEqual(image_response.status, 404)

    def test_default_server_loads_hf_token_from_env_file_before_model_runner(self):
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

            with patch.dict(os.environ, {}, clear=True):
                with patch("catbox.browser_ui.SdTurboImageToImageModelRunner", TokenAwareRunner):
                    with patch("catbox.browser_ui.CatboxBrowserUiServer") as server_class:
                        make_default_browser_ui_server(
                            ("127.0.0.1", 0),
                            env_file=env_file,
                        )
                        self.assertEqual(server_class.call_count, 1)

            self.assertEqual(observed_token, ["hf_test_token"])
