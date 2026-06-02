from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, urlparse

from catbox.model_backend import CatboxModelBackend
from catbox.sd_turbo_runner import DEFAULT_RUNTIME_DIR, SdTurboImageToImageModelRunner


class BrowserBackend(Protocol):
    def readiness(self) -> dict[str, str]: ...

    def observe(self) -> dict[str, object]: ...


class BrowserUiResponse:
    def __init__(
        self,
        status: int,
        body: bytes,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self.body = body
        self.headers = headers or {}


class BrowserUiApp:
    def __init__(self, backend: BrowserBackend) -> None:
        self._backend = backend
        self._generated_image_refs: set[str] = set()

    def handle(self, method: str, path: str) -> BrowserUiResponse:
        parsed = urlparse(path)
        if method == "GET" and parsed.path == "/":
            return self._page()
        if method == "GET" and parsed.path == "/api/readiness":
            return self._readiness()
        if method == "POST" and parsed.path == "/api/observe":
            return self._observe()
        if method == "GET" and parsed.path == "/api/generated-outcome":
            return self._generated_outcome(parsed.query)
        return BrowserUiResponse(404, b"Not found", {"Content-Type": "text/plain; charset=utf-8"})

    def _page(self) -> BrowserUiResponse:
        return BrowserUiResponse(
            200,
            _BROWSER_UI_HTML.encode("utf-8"),
            {"Content-Type": "text/html; charset=utf-8"},
        )

    def _readiness(self) -> BrowserUiResponse:
        response_bytes = json.dumps(self._backend.readiness()).encode("utf-8")
        return BrowserUiResponse(
            200,
            response_bytes,
            {"Content-Type": "application/json"},
        )

    def _observe(self) -> BrowserUiResponse:
        response = self._backend.observe()
        if response.get("status") == "generated" and isinstance(response.get("imageRef"), str):
            self._generated_image_refs.add(response["imageRef"])
        response_bytes = json.dumps(response).encode("utf-8")
        return BrowserUiResponse(
            200,
            response_bytes,
            {"Content-Type": "application/json"},
        )

    def _generated_outcome(self, query: str) -> BrowserUiResponse:
        image_ref = parse_qs(query).get("imageRef", [""])[0]
        if image_ref not in self._generated_image_refs:
            return BrowserUiResponse(404, b"Not found", {"Content-Type": "text/plain; charset=utf-8"})

        image_path = Path(image_ref)
        if not image_path.is_file():
            return BrowserUiResponse(404, b"Not found", {"Content-Type": "text/plain; charset=utf-8"})

        content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        return BrowserUiResponse(
            200,
            image_path.read_bytes(),
            {"Content-Type": content_type},
        )


class CatboxBrowserUiServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], backend: BrowserBackend) -> None:
        self.app = BrowserUiApp(backend)
        super().__init__(server_address, CatboxBrowserUiRequestHandler)


class CatboxBrowserUiRequestHandler(BaseHTTPRequestHandler):
    server: CatboxBrowserUiServer

    def do_POST(self) -> None:
        self._send_app_response(self.server.app.handle("POST", self.path))

    def do_GET(self) -> None:
        self._send_app_response(self.server.app.handle("GET", self.path))

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_app_response(self, response: BrowserUiResponse) -> None:
        self.send_response(response.status)
        for header, value in response.headers.items():
            self.send_header(header, value)
        self.send_header("Content-Length", str(len(response.body)))
        self.end_headers()
        self.wfile.write(response.body)


def make_browser_ui_server(
    server_address: tuple[str, int],
    backend: BrowserBackend,
) -> CatboxBrowserUiServer:
    return CatboxBrowserUiServer(server_address, backend)


def make_default_browser_ui_server(
    server_address: tuple[str, int],
    runtime_dir: str | Path = DEFAULT_RUNTIME_DIR,
) -> CatboxBrowserUiServer:
    backend = CatboxModelBackend(
        model_runner=SdTurboImageToImageModelRunner(runtime_dir=runtime_dir)
    )
    return make_browser_ui_server(server_address, backend)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Catbox Browser UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--runtime-dir", default=str(DEFAULT_RUNTIME_DIR))
    args = parser.parse_args()

    server = make_default_browser_ui_server(
        (args.host, args.port),
        runtime_dir=args.runtime_dir,
    )
    print(f"Catbox Browser UI listening on http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


_BROWSER_UI_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Catbox</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #171717;
      color: #f5f0e8;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background:
        radial-gradient(circle at 50% 18%, rgba(198, 85, 70, 0.2), transparent 28rem),
        linear-gradient(145deg, #161616, #22201d 54%, #101010);
    }

    main {
      width: min(92vw, 760px);
      display: grid;
      gap: 24px;
      justify-items: center;
      text-align: center;
    }

    .stage {
      width: min(88vw, 560px);
      aspect-ratio: 1.18;
      display: grid;
      place-items: center;
      border: 1px solid rgba(245, 240, 232, 0.14);
      background: rgba(255, 255, 255, 0.035);
      overflow: hidden;
    }

    .box {
      width: min(58vw, 300px);
      aspect-ratio: 1.25;
      position: relative;
      border: 2px solid #d8aa72;
      background: linear-gradient(#9f6a38, #6e431f);
      box-shadow: 0 26px 60px rgba(0, 0, 0, 0.45);
    }

    .box::before {
      content: "";
      position: absolute;
      inset: -44px -16px auto;
      height: 54px;
      border: 2px solid #e0b782;
      background: linear-gradient(#bb8447, #7c4d25);
      transform: perspective(260px) rotateX(46deg);
      transform-origin: bottom;
    }

    .panel {
      display: none;
      width: 100%;
      justify-items: center;
      gap: 18px;
    }

    .panel.is-active {
      display: grid;
    }

    button {
      min-width: 148px;
      min-height: 44px;
      border: 1px solid rgba(245, 240, 232, 0.2);
      background: #f5f0e8;
      color: #191817;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }

    button:disabled {
      opacity: 0.55;
      cursor: progress;
    }

    .actions {
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: 12px;
    }

    .noise {
      width: min(72vw, 420px);
      aspect-ratio: 1.4;
      background:
        repeating-radial-gradient(circle at 20% 30%, rgba(255, 255, 255, 0.18) 0 1px, transparent 1px 5px),
        repeating-linear-gradient(112deg, rgba(237, 191, 115, 0.2) 0 2px, transparent 2px 8px),
        #202020;
      animation: pulse 900ms steps(2, end) infinite;
    }

    @keyframes pulse {
      from { filter: contrast(1); }
      to { filter: contrast(1.8) brightness(1.08); }
    }

    img {
      width: min(88vw, 560px);
      aspect-ratio: 1;
      object-fit: contain;
      background: #101010;
      border: 1px solid rgba(245, 240, 232, 0.16);
    }

    .meta,
    .note {
      margin: 0;
      max-width: 54ch;
      line-height: 1.55;
      color: rgba(245, 240, 232, 0.76);
    }

    .failure-title {
      margin: 0;
      font-size: clamp(1.35rem, 5vw, 2rem);
      line-height: 1.1;
    }
  </style>
</head>
<body>
  <main>
    <section class="panel is-active" data-state="starting" id="starting-panel" aria-live="polite">
      <div class="stage" aria-label="Catbox model backend startup">
        <div class="box"></div>
      </div>
      <p class="note">Preparing the model backend</p>
    </section>

    <section class="panel" data-state="sealed" id="sealed-panel" aria-live="polite">
      <div class="stage" aria-label="A sealed Catbox">
        <div class="box"></div>
      </div>
      <button id="observe-button" type="button">Observe</button>
    </section>

    <section class="panel" data-state="waiting" id="waiting-panel" aria-live="polite">
      <div class="noise" aria-hidden="true"></div>
      <p class="note">Observation noise</p>
      <p class="note" id="progressive-waiting-status" hidden>The model backend is still generating this observation.</p>
    </section>

    <section class="panel" data-state="revealed" id="revealed-panel" aria-live="polite">
      <img id="generated-outcome" alt="Generated Outcome">
      <p class="meta" id="outcome-metadata"></p>
      <p class="note" id="reveal-note"></p>
      <button id="reset-button" type="button">Reset</button>
    </section>

    <section class="panel" data-state="generation-failure" id="generation-failure-panel" aria-live="assertive">
      <div class="stage" aria-label="Generation Failure">
        <div>
          <h1 class="failure-title">Generation Failure</h1>
          <p class="note" id="generation-failure-message"></p>
        </div>
      </div>
      <div class="actions">
        <button id="retry-button" type="button">Retry</button>
        <button id="failure-reset-button" type="button">Reset</button>
      </div>
    </section>
  </main>

  <script>
    const startingPanel = document.querySelector("#starting-panel");
    const sealedPanel = document.querySelector("#sealed-panel");
    const waitingPanel = document.querySelector("#waiting-panel");
    const revealedPanel = document.querySelector("#revealed-panel");
    const generationFailurePanel = document.querySelector("#generation-failure-panel");
    const observeButton = document.querySelector("#observe-button");
    const resetButton = document.querySelector("#reset-button");
    const retryButton = document.querySelector("#retry-button");
    const failureResetButton = document.querySelector("#failure-reset-button");
    const generatedOutcome = document.querySelector("#generated-outcome");
    const revealNote = document.querySelector("#reveal-note");
    const outcomeMetadata = document.querySelector("#outcome-metadata");
    const generationFailureMessage = document.querySelector("#generation-failure-message");
    const progressiveWaitingStatus = document.querySelector("#progressive-waiting-status");
    const PROGRESSIVE_WAITING_DELAY_MS = 4500;
    let progressiveWaitingTimer = null;

    function show(panel) {
      for (const candidate of [startingPanel, sealedPanel, waitingPanel, revealedPanel, generationFailurePanel]) {
        candidate.classList.toggle("is-active", candidate === panel);
      }
    }

    function clearProgressiveWaiting() {
      if (progressiveWaitingTimer !== null) {
        window.clearTimeout(progressiveWaitingTimer);
        progressiveWaitingTimer = null;
      }
      progressiveWaitingStatus.hidden = true;
    }

    function showProgressiveWaiting() {
      progressiveWaitingStatus.hidden = false;
      progressiveWaitingTimer = null;
    }

    function startProgressiveWaitingTimer() {
      clearProgressiveWaiting();
      progressiveWaitingTimer = window.setTimeout(
        showProgressiveWaiting,
        PROGRESSIVE_WAITING_DELAY_MS
      );
    }

    async function refreshReadiness() {
      const response = await fetch("/api/readiness");
      const readiness = await response.json();
      if (readiness.status === "ready") {
        observeButton.disabled = false;
        show(sealedPanel);
        return;
      }
      observeButton.disabled = true;
      show(startingPanel);
      window.setTimeout(refreshReadiness, 1200);
    }

    function resetToSealed() {
      clearProgressiveWaiting();
      generatedOutcome.removeAttribute("src");
      outcomeMetadata.textContent = "";
      revealNote.textContent = "";
      generationFailureMessage.textContent = "";
      observeButton.disabled = false;
      retryButton.disabled = false;
      show(sealedPanel);
    }

    function showGenerationFailure(observation) {
      clearProgressiveWaiting();
      const message = observation?.error?.message || "The model backend could not produce a Generated Outcome.";
      generationFailureMessage.textContent = message;
      observeButton.disabled = false;
      retryButton.disabled = false;
      show(generationFailurePanel);
    }

    async function observe() {
      observeButton.disabled = true;
      retryButton.disabled = true;
      show(waitingPanel);
      startProgressiveWaitingTimer();

      let observation;
      try {
        const response = await fetch("/api/observe", { method: "POST" });
        observation = await response.json();
      } catch (error) {
        showGenerationFailure({
          error: {
            message: "The model backend could not be reached.",
          },
        });
        return;
      }

      if (observation.status === "generation_failed") {
        showGenerationFailure(observation);
        return;
      }
      if (observation.status !== "generated") {
        showGenerationFailure({
          error: {
            message: "The model backend returned an unknown observation state.",
          },
        });
        return;
      }

      generatedOutcome.src = `/api/generated-outcome?imageRef=${encodeURIComponent(observation.imageRef)}`;
      outcomeMetadata.textContent = `Outcome: ${observation.outcome}`;
      revealNote.textContent = observation.revealNote;
      clearProgressiveWaiting();
      show(revealedPanel);
    }

    observeButton.addEventListener("click", observe);
    retryButton.addEventListener("click", observe);
    resetButton.addEventListener("click", resetToSealed);
    failureResetButton.addEventListener("click", resetToSealed);

    refreshReadiness();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
