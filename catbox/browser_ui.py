from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, urlparse

from catbox.env_loader import DEFAULT_ENV_FILE, load_env_file
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
    env_file: str | Path = DEFAULT_ENV_FILE,
) -> CatboxBrowserUiServer:
    load_env_file(env_file)
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
      --bg: #131313;
      --surface: #1d1b19;
      --surface-soft: rgba(255, 255, 255, 0.055);
      --text: #f6efe4;
      --muted: rgba(246, 239, 228, 0.66);
      --border: rgba(246, 239, 228, 0.16);
      --amber: #d8a05f;
      --amber-strong: #f1bc74;
      --teal: #6dc9bd;
      --danger: #e46f5f;
      --ease-out: cubic-bezier(0.23, 1, 0.32, 1);
      --ease-in-out: cubic-bezier(0.77, 0, 0.175, 1);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 28px 18px;
      background:
        radial-gradient(circle at 22% 12%, rgba(109, 201, 189, 0.18), transparent 24rem),
        radial-gradient(circle at 78% 16%, rgba(228, 111, 95, 0.16), transparent 28rem),
        linear-gradient(145deg, #121212 0%, #211c18 58%, #101010 100%);
    }

    main {
      width: min(94vw, 780px);
      display: grid;
      gap: 18px;
      justify-items: center;
      text-align: center;
    }

    .brand {
      display: grid;
      gap: 6px;
      justify-items: center;
    }

    h1 {
      margin: 0;
      font-size: clamp(2.35rem, 10vw, 5.5rem);
      font-weight: 850;
      line-height: 0.88;
      letter-spacing: 0;
    }

    .tagline {
      margin: 0;
      max-width: 36rem;
      color: var(--muted);
      font-size: clamp(0.95rem, 2.6vw, 1.08rem);
      line-height: 1.45;
    }

    .viewport {
      width: 100%;
      min-height: min(86vh, 690px);
      display: grid;
      place-items: center;
      position: relative;
    }

    .stage {
      width: min(88vw, 620px);
      aspect-ratio: 1.12;
      display: grid;
      place-items: center;
      position: relative;
      border: 1px solid var(--border);
      background:
        radial-gradient(circle at 50% 32%, rgba(246, 239, 228, 0.09), transparent 18rem),
        linear-gradient(180deg, rgba(255, 255, 255, 0.055), rgba(255, 255, 255, 0.025));
      overflow: hidden;
      box-shadow: 0 34px 90px rgba(0, 0, 0, 0.34);
    }

    .stage::before {
      content: "";
      position: absolute;
      inset: auto 0 0;
      height: 34%;
      background: linear-gradient(180deg, transparent, rgba(216, 160, 95, 0.14));
      pointer-events: none;
    }

    .stage::after {
      content: "";
      position: absolute;
      width: 70%;
      height: 1px;
      bottom: 28%;
      background: linear-gradient(90deg, transparent, rgba(246, 239, 228, 0.2), transparent);
      pointer-events: none;
    }

    .box {
      width: min(58vw, 330px);
      aspect-ratio: 1.25;
      position: relative;
      transform-style: preserve-3d;
      transform: translateY(12px) perspective(760px) rotateX(2deg);
      border: 2px solid rgba(241, 188, 116, 0.78);
      background:
        linear-gradient(115deg, rgba(255, 255, 255, 0.11), transparent 34%),
        linear-gradient(#9f6a38, #6d421f);
      box-shadow:
        0 34px 70px rgba(0, 0, 0, 0.44),
        inset 0 1px 0 rgba(255, 255, 255, 0.16);
      transition: transform 240ms var(--ease-out);
    }

    .box::before {
      content: "";
      position: absolute;
      inset: -52px -18px auto;
      height: 62px;
      border: 2px solid rgba(241, 188, 116, 0.88);
      background:
        linear-gradient(110deg, rgba(255, 255, 255, 0.14), transparent 42%),
        linear-gradient(#bf884c, #7d4d25);
      transform: perspective(280px) rotateX(48deg);
      transform-origin: bottom;
      box-shadow: 0 18px 24px rgba(0, 0, 0, 0.18);
      transition: transform 320ms var(--ease-in-out), inset 320ms var(--ease-in-out);
    }

    .box::after {
      content: "";
      position: absolute;
      inset: 22% 18% auto;
      height: 26%;
      background: radial-gradient(ellipse, rgba(0, 0, 0, 0.22), transparent 66%);
      transform: translateY(16px);
      opacity: 0.64;
    }

    .panel {
      position: absolute;
      inset: 0;
      width: 100%;
      min-height: 100%;
      display: grid;
      align-content: center;
      justify-items: center;
      gap: 18px;
      opacity: 0;
      pointer-events: none;
      transform: translateY(10px) scale(0.985);
      filter: blur(2px);
      transition:
        opacity 190ms var(--ease-out),
        transform 220ms var(--ease-out),
        filter 220ms var(--ease-out);
    }

    .panel.is-active {
      position: relative;
      display: grid;
      opacity: 1;
      pointer-events: auto;
      transform: translateY(0) scale(1);
      filter: blur(0);
    }

    button {
      min-width: 148px;
      min-height: 46px;
      border: 1px solid rgba(246, 239, 228, 0.28);
      background: linear-gradient(180deg, #fff6e7, #e6c99f);
      color: #1b1714;
      font: inherit;
      font-weight: 700;
      border-radius: 8px;
      cursor: pointer;
      box-shadow:
        0 14px 34px rgba(0, 0, 0, 0.28),
        inset 0 1px 0 rgba(255, 255, 255, 0.5);
      transition:
        transform 140ms var(--ease-out),
        box-shadow 180ms var(--ease-out),
        border-color 180ms ease,
        opacity 180ms ease;
    }

    button:active {
      transform: scale(0.97);
    }

    button:disabled {
      opacity: 0.55;
      cursor: progress;
      box-shadow: none;
    }

    @media (hover: hover) and (pointer: fine) {
      button:not(:disabled):hover {
        transform: translateY(-1px);
        border-color: rgba(246, 239, 228, 0.48);
        box-shadow:
          0 18px 40px rgba(0, 0, 0, 0.32),
          inset 0 1px 0 rgba(255, 255, 255, 0.58);
      }

      button:not(:disabled):hover:active {
        transform: translateY(0) scale(0.97);
      }
    }

    .actions {
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: 12px;
    }

    .noise {
      width: min(78vw, 520px);
      aspect-ratio: 1.18;
      position: relative;
      overflow: hidden;
      border: 1px solid var(--border);
      background:
        radial-gradient(circle at 50% 46%, rgba(109, 201, 189, 0.18), transparent 15rem),
        repeating-linear-gradient(102deg, rgba(241, 188, 116, 0.18) 0 1px, transparent 1px 8px),
        repeating-radial-gradient(circle at 26% 24%, rgba(246, 239, 228, 0.16) 0 1px, transparent 1px 5px),
        #171717;
      box-shadow: 0 30px 78px rgba(0, 0, 0, 0.36);
      animation: observation-pulse 760ms steps(2, end) infinite;
    }

    .noise::before,
    .noise::after {
      content: "";
      position: absolute;
      inset: 18%;
      border: 1px solid rgba(246, 239, 228, 0.22);
      transform: rotate(45deg) scale(0.84);
      animation: aperture 1600ms var(--ease-in-out) infinite alternate;
    }

    .noise::after {
      inset: 29%;
      border-color: rgba(109, 201, 189, 0.34);
      animation-delay: 180ms;
    }

    @keyframes observation-pulse {
      from { filter: contrast(1); }
      to { filter: contrast(1.75) brightness(1.08); }
    }

    @keyframes aperture {
      from { transform: rotate(45deg) scale(0.78); opacity: 0.38; }
      to { transform: rotate(45deg) scale(1.04); opacity: 0.74; }
    }

    img {
      width: min(88vw, 620px);
      aspect-ratio: 1;
      object-fit: contain;
      background: #111;
      border: 1px solid var(--border);
      box-shadow: 0 32px 86px rgba(0, 0, 0, 0.38);
      opacity: 1;
      transform: scale(1);
      transition:
        opacity 240ms var(--ease-out),
        transform 280ms var(--ease-out),
        filter 280ms var(--ease-out);
    }

    img[data-loading="true"] {
      opacity: 0;
      transform: scale(0.975);
      filter: blur(4px);
    }

    .meta,
    .note {
      margin: 0;
      max-width: 54ch;
      line-height: 1.55;
      color: var(--muted);
    }

    .meta {
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 5px 10px;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: var(--surface-soft);
      color: rgba(246, 239, 228, 0.78);
      font-size: 0.88rem;
    }

    [hidden] {
      display: none !important;
    }

    .failure-title {
      margin: 0;
      font-size: clamp(1.35rem, 5vw, 2rem);
      line-height: 1.1;
    }

    #generation-failure-panel .stage {
      border-color: rgba(228, 111, 95, 0.34);
      background:
        radial-gradient(circle at 50% 38%, rgba(228, 111, 95, 0.17), transparent 16rem),
        linear-gradient(180deg, rgba(255, 255, 255, 0.055), rgba(255, 255, 255, 0.025));
    }

    @media (max-width: 560px) {
      body {
        padding: 20px 12px;
      }

      main {
        gap: 14px;
      }

      .viewport {
        min-height: min(82vh, 610px);
      }

      .stage,
      img {
        width: min(94vw, 620px);
      }

      .box {
        width: min(66vw, 300px);
      }
    }

    @media (prefers-reduced-motion: reduce) {
      *,
      *::before,
      *::after {
        animation-duration: 1ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: 1ms !important;
      }

      .panel,
      img[data-loading="true"] {
        transform: none;
        filter: none;
      }
    }
  </style>
</head>
<body>
  <main>
    <header class="brand">
      <h1>Catbox</h1>
      <p class="tagline">Observe the sealed box. The local model resolves one generated outcome.</p>
    </header>

    <div class="viewport">
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
        <img id="generated-outcome" alt="Generated Outcome" data-loading="true">
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
    </div>
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
      generatedOutcome.dataset.loading = "true";
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

      generatedOutcome.dataset.loading = "true";
      outcomeMetadata.textContent = `Outcome: ${observation.outcome}`;
      revealNote.textContent = observation.revealNote;
      generatedOutcome.onload = () => {
        generatedOutcome.dataset.loading = "false";
        clearProgressiveWaiting();
        show(revealedPanel);
      };
      generatedOutcome.onerror = () => {
        showGenerationFailure({
          error: {
            message: "The generated outcome file could not be displayed.",
          },
        });
      };
      generatedOutcome.src = `/api/generated-outcome?imageRef=${encodeURIComponent(observation.imageRef)}`;
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
