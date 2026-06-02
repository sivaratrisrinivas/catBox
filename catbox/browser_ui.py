from __future__ import annotations

import argparse
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, urlparse

from catbox.env_loader import DEFAULT_ENV_FILE, load_env_file
from catbox.model_backend import CatboxModelBackend
from catbox.sd_turbo_runner import DEFAULT_RUNTIME_DIR, SdTurboImageToImageModelRunner


class BrowserBackend(Protocol):
    def readiness(self) -> dict[str, str]: ...

    def observe(self, trace_callback=None) -> dict[str, object]: ...


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
        self._trace_frame_refs: list[str] = []
        self._lock = threading.Lock()

    def handle(self, method: str, path: str) -> BrowserUiResponse:
        parsed = urlparse(path)
        if method == "GET" and parsed.path == "/":
            return self._page()
        if method == "GET" and parsed.path == "/api/readiness":
            return self._readiness()
        if method == "POST" and parsed.path == "/api/observe":
            return self._observe()
        if method == "GET" and parsed.path == "/api/trace":
            return self._trace()
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
        with self._lock:
            self._trace_frame_refs = []

        def register_trace_frame(image_ref: str) -> None:
            with self._lock:
                self._trace_frame_refs.append(image_ref)
                self._generated_image_refs.add(image_ref)

        response = self._backend.observe(trace_callback=register_trace_frame)
        if response.get("status") == "generated" and isinstance(response.get("imageRef"), str):
            with self._lock:
                self._generated_image_refs.add(response["imageRef"])
                trace_refs = response.get("traceRefs")
                if isinstance(trace_refs, list):
                    for trace_ref in trace_refs:
                        if isinstance(trace_ref, str):
                            self._generated_image_refs.add(trace_ref)
        response_bytes = json.dumps(response).encode("utf-8")
        return BrowserUiResponse(
            200,
            response_bytes,
            {"Content-Type": "application/json"},
        )

    def _trace(self) -> BrowserUiResponse:
        with self._lock:
            response = {"traceRefs": list(self._trace_frame_refs)}
        return BrowserUiResponse(
            200,
            json.dumps(response).encode("utf-8"),
            {"Content-Type": "application/json"},
        )

    def _generated_outcome(self, query: str) -> BrowserUiResponse:
        image_ref = parse_qs(query).get("imageRef", [""])[0]
        with self._lock:
            image_is_registered = image_ref in self._generated_image_refs
        if not image_is_registered:
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
      --void: #070807;
      --ink: #f4ead6;
      --muted: rgba(244, 234, 214, 0.64);
      --faint: rgba(244, 234, 214, 0.18);
      --brass: #d2a460;
      --cyan: #75d6ce;
      --red: #e16652;
      --ease-out: cubic-bezier(0.23, 1, 0.32, 1);
      font-family: "Georgia", "Times New Roman", serif;
      background: var(--void);
      color: var(--ink);
    }

    * {
      box-sizing: border-box;
    }

    html {
      scroll-behavior: smooth;
    }

    body {
      margin: 0;
      min-height: 100vh;
      overflow-x: hidden;
      overflow-y: auto;
      background:
        linear-gradient(rgba(244, 234, 214, 0.028) 1px, transparent 1px),
        linear-gradient(90deg, rgba(244, 234, 214, 0.026) 1px, transparent 1px),
        linear-gradient(135deg, #090a09 0%, #15120d 48%, #050606 100%);
      background-size: 44px 44px, 44px 44px, auto;
    }

    main {
      min-height: 100svh;
      position: relative;
      isolation: isolate;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      gap: 16px;
      padding: 24px 28px 28px;
    }

    .brand {
      position: relative;
      z-index: 3;
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 18px;
    }

    h1 {
      margin: 0;
      font-size: clamp(2rem, 6vw, 5.6rem);
      font-weight: 500;
      line-height: 0.86;
      letter-spacing: 0;
      text-transform: uppercase;
    }

    .tagline {
      margin: 0;
      max-width: 31rem;
      color: var(--muted);
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: clamp(0.72rem, 1.3vw, 0.9rem);
      line-height: 1.55;
      text-align: right;
      text-transform: uppercase;
    }

    .viewport {
      width: 100%;
      min-height: 0;
      position: relative;
    }

    .panel {
      display: none;
      place-items: center;
      min-height: clamp(320px, calc(100svh - 220px), 700px);
      opacity: 0;
      pointer-events: none;
      transition:
        opacity 180ms var(--ease-out);
    }

    .panel.is-active {
      display: grid;
      opacity: 1;
      pointer-events: auto;
    }

    .stage {
      position: relative;
      width: 100%;
      min-height: inherit;
      display: grid;
      place-items: center;
      overflow: hidden;
    }

    .box {
      width: min(52vw, 430px);
      aspect-ratio: 1.25;
      position: relative;
      z-index: 1;
      transform: translateY(24px) perspective(900px) rotateX(4deg) rotateZ(-1deg);
      background:
        linear-gradient(115deg, rgba(255, 255, 255, 0.16), transparent 32%),
        linear-gradient(180deg, #b67b3a 0%, #77451f 100%);
      border: 2px solid rgba(244, 190, 104, 0.9);
      box-shadow:
        0 38px 82px rgba(0, 0, 0, 0.58),
        inset 0 1px 0 rgba(255, 255, 255, 0.22);
    }

    .box::before {
      content: "";
      position: absolute;
      inset: -68px -22px auto;
      height: 78px;
      background:
        linear-gradient(110deg, rgba(255, 255, 255, 0.18), transparent 45%),
        linear-gradient(180deg, #c8924d 0%, #825126 100%);
      border: 2px solid rgba(244, 190, 104, 0.95);
      transform: perspective(320px) rotateX(52deg);
      transform-origin: bottom;
      box-shadow: 0 24px 28px rgba(0, 0, 0, 0.24);
    }

    .box::after {
      content: "";
      position: absolute;
      inset: 24% 19% auto;
      height: 30%;
      background: radial-gradient(ellipse, rgba(0, 0, 0, 0.3), transparent 68%);
      transform: translateY(22px);
    }

    .box-label {
      position: absolute;
      z-index: 1;
      left: 20px;
      bottom: 16px;
      color: rgba(244, 234, 214, 0.74);
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0;
      text-transform: uppercase;
    }

    .trace-surface,
    .outcome-surface {
      width: min(88vw, 58svh, 620px);
      aspect-ratio: 1;
      position: relative;
      display: grid;
      place-items: center;
    }

    .trace-surface::before,
    .outcome-surface::before {
      content: "";
      position: absolute;
      inset: -16px;
      border-top: 1px solid rgba(210, 164, 96, 0.48);
      border-left: 1px solid rgba(117, 214, 206, 0.34);
      border-right: 1px solid rgba(244, 234, 214, 0.1);
      pointer-events: none;
    }

    .trace-placeholder {
      position: absolute;
      inset: 0;
      background:
        repeating-linear-gradient(104deg, rgba(210, 164, 96, 0.18) 0 1px, transparent 1px 10px),
        repeating-radial-gradient(circle at 28% 22%, rgba(117, 214, 206, 0.16) 0 1px, transparent 1px 5px),
        radial-gradient(circle at 50% 50%, rgba(117, 214, 206, 0.16), transparent 28rem),
        #0d0f0e;
      animation: trace-static 620ms steps(2, end) infinite;
    }

    @keyframes trace-static {
      from { filter: contrast(1.05) brightness(0.92); }
      to { filter: contrast(1.8) brightness(1.08); }
    }

    img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      opacity: 1;
      transform: scale(1);
      filter: contrast(1.02);
      transition:
        opacity 220ms var(--ease-out),
        transform 260ms var(--ease-out),
        filter 260ms var(--ease-out);
    }

    img[data-loading="true"],
    #trace-frame[data-empty="true"] {
      opacity: 0;
      transform: scale(0.985);
      filter: blur(8px);
    }

    #trace-frame {
      position: relative;
      z-index: 1;
      image-rendering: auto;
      mix-blend-mode: screen;
    }

    #generated-outcome {
      position: relative;
      z-index: 1;
    }

    .hud {
      position: relative;
      z-index: 4;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: end;
      gap: 20px;
      pointer-events: auto;
    }

    .note {
      margin: 0;
      max-width: 58ch;
      color: var(--muted);
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 0.8rem;
      line-height: 1.55;
      text-transform: uppercase;
    }

    .meta {
      margin: 0;
      color: rgba(244, 234, 214, 0.84);
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 0.78rem;
      line-height: 1.55;
      text-transform: uppercase;
    }

    .actions {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 10px;
      pointer-events: auto;
    }

    button {
      min-width: 132px;
      min-height: 44px;
      border: 1px solid rgba(244, 234, 214, 0.35);
      background: rgba(244, 234, 214, 0.92);
      color: #0b0b09;
      font: inherit;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 0.78rem;
      font-weight: 800;
      letter-spacing: 0;
      text-transform: uppercase;
      cursor: pointer;
      box-shadow: 0 18px 42px rgba(0, 0, 0, 0.34);
      transition:
        transform 140ms var(--ease-out),
        background-color 180ms ease,
        border-color 180ms ease,
        opacity 180ms ease;
    }

    button:active {
      transform: scale(0.97);
    }

    button:disabled {
      opacity: 0.48;
      cursor: progress;
      box-shadow: none;
    }

    .secondary-button {
      background: rgba(7, 8, 7, 0.34);
      color: var(--ink);
    }

    @media (hover: hover) and (pointer: fine) {
      button:not(:disabled):hover {
        transform: translateY(-1px);
        border-color: rgba(117, 214, 206, 0.7);
      }

      button:not(:disabled):hover:active {
        transform: translateY(0) scale(0.97);
      }
    }

    .failure-title {
      margin: 0;
      color: var(--red);
      font-size: clamp(2rem, 7vw, 6rem);
      font-weight: 500;
      line-height: 0.88;
      letter-spacing: 0;
      text-transform: uppercase;
    }

    .failure-stack {
      display: grid;
      gap: 16px;
      justify-items: center;
      text-align: center;
      padding: 0 28px;
    }

    [hidden] {
      display: none !important;
    }

    @media (max-width: 720px) {
      main {
        grid-template-rows: auto minmax(0, 1fr) auto;
        gap: 12px;
        padding: 18px 16px 20px;
      }

      .brand {
        display: grid;
        gap: 8px;
      }

      .tagline {
        text-align: left;
      }

      .hud {
        grid-template-columns: 1fr;
      }

      .actions {
        justify-content: stretch;
      }

      button {
        flex: 1;
      }

      .box {
        width: min(72vw, 360px);
      }

      .trace-surface,
      .outcome-surface {
        width: min(94vw, 50svh, 480px);
      }

      .panel {
        min-height: clamp(280px, calc(100svh - 250px), 560px);
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

      img[data-loading="true"] {
        filter: none;
      }
    }
  </style>
</head>
<body>
  <main>
    <header class="brand">
      <h1>Catbox</h1>
      <p class="tagline">A sealed system, one entropy seed, one real denoising trace.</p>
    </header>

    <div class="viewport">
      <section class="panel is-active" data-state="starting" id="starting-panel" aria-live="polite">
        <div class="stage" aria-label="Catbox model backend startup">
          <div class="box"><div class="box-label">preloading</div></div>
        </div>
      </section>

      <section class="panel" data-state="sealed" id="sealed-panel" aria-live="polite">
        <div class="stage" aria-label="A sealed Catbox">
          <div class="box"><div class="box-label">unobserved</div></div>
        </div>
      </section>

      <section class="panel" data-state="waiting" id="waiting-panel" aria-live="polite">
        <div class="stage" aria-label="Captured Denoising Trace">
          <div class="trace-surface">
            <div class="trace-placeholder" id="trace-placeholder"></div>
            <img id="trace-frame" alt="Captured Denoising Trace frame" data-empty="true">
          </div>
        </div>
      </section>

      <section class="panel" data-state="revealed" id="revealed-panel" aria-live="polite">
        <div class="stage" aria-label="Generated Outcome">
          <div class="outcome-surface">
            <img id="generated-outcome" alt="Generated Outcome" data-loading="true">
          </div>
        </div>
      </section>

      <section class="panel" data-state="generation-failure" id="generation-failure-panel" aria-live="assertive">
        <div class="stage" aria-label="Generation Failure">
          <div class="failure-stack">
            <h2 class="failure-title">Generation Failure</h2>
            <p class="note" id="generation-failure-message"></p>
          </div>
        </div>
      </section>
    </div>

    <div class="hud">
      <div>
        <p class="meta" id="outcome-metadata"></p>
        <p class="note" id="system-status">Preparing the local diffusion chamber</p>
        <p class="note" id="progressive-waiting-status" hidden>The model is still resolving this observation.</p>
        <p class="note" id="reveal-note"></p>
      </div>
      <div class="actions">
        <button id="observe-button" type="button" disabled>Observe</button>
        <button class="secondary-button" id="reset-button" type="button">Reset</button>
        <button id="retry-button" type="button" hidden>Retry</button>
        <button class="secondary-button" id="failure-reset-button" type="button" hidden>Reset</button>
      </div>
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
    const traceFrame = document.querySelector("#trace-frame");
    const tracePlaceholder = document.querySelector("#trace-placeholder");
    const revealNote = document.querySelector("#reveal-note");
    const outcomeMetadata = document.querySelector("#outcome-metadata");
    const systemStatus = document.querySelector("#system-status");
    const generationFailureMessage = document.querySelector("#generation-failure-message");
    const progressiveWaitingStatus = document.querySelector("#progressive-waiting-status");
    const PROGRESSIVE_WAITING_DELAY_MS = 4500;
    const TRACE_POLL_MS = 360;
    let progressiveWaitingTimer = null;
    let tracePollTimer = null;
    let lastTraceRef = "";

    function show(panel) {
      for (const candidate of [startingPanel, sealedPanel, waitingPanel, revealedPanel, generationFailurePanel]) {
        candidate.classList.toggle("is-active", candidate === panel);
      }
    }

    function bringActivePanelIntoView(panel) {
      window.requestAnimationFrame(() => {
        panel.scrollIntoView({
          behavior: "smooth",
          block: "center",
          inline: "nearest",
        });
      });
    }

    function setFailureActions(isFailure) {
      retryButton.hidden = !isFailure;
      failureResetButton.hidden = !isFailure;
      observeButton.hidden = isFailure;
      resetButton.hidden = isFailure;
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

    function clearTracePolling() {
      if (tracePollTimer !== null) {
        window.clearTimeout(tracePollTimer);
        tracePollTimer = null;
      }
    }

    function clearTraceFrame() {
      lastTraceRef = "";
      traceFrame.removeAttribute("src");
      traceFrame.dataset.empty = "true";
      tracePlaceholder.hidden = false;
    }

    async function pollTrace() {
      try {
        const response = await fetch("/api/trace");
        const payload = await response.json();
        const frames = Array.isArray(payload.traceRefs) ? payload.traceRefs : [];
        const latest = frames[frames.length - 1];
        if (latest && latest !== lastTraceRef) {
          lastTraceRef = latest;
          traceFrame.dataset.empty = "false";
          tracePlaceholder.hidden = true;
          traceFrame.src = `/api/generated-outcome?imageRef=${encodeURIComponent(latest)}`;
          systemStatus.textContent = `Captured Denoising Trace frame ${frames.length}`;
        }
      } catch (error) {
        systemStatus.textContent = "Waiting for trace frames from the Model Backend";
      }
      tracePollTimer = window.setTimeout(pollTrace, TRACE_POLL_MS);
    }

    async function refreshReadiness() {
      const response = await fetch("/api/readiness");
      const readiness = await response.json();
      if (readiness.status === "ready") {
        observeButton.disabled = false;
        resetButton.disabled = false;
        systemStatus.textContent = "System sealed. Ready for observation.";
        show(sealedPanel);
        return;
      }
      observeButton.disabled = true;
      resetButton.disabled = true;
      systemStatus.textContent = "Preparing the local diffusion chamber";
      show(startingPanel);
      window.setTimeout(refreshReadiness, 1200);
    }

    function resetToSealed() {
      clearProgressiveWaiting();
      clearTracePolling();
      clearTraceFrame();
      generatedOutcome.removeAttribute("src");
      generatedOutcome.dataset.loading = "true";
      outcomeMetadata.textContent = "";
      revealNote.textContent = "";
      generationFailureMessage.textContent = "";
      systemStatus.textContent = "System sealed. Ready for observation.";
      observeButton.disabled = false;
      retryButton.disabled = false;
      setFailureActions(false);
      show(sealedPanel);
    }

    function showGenerationFailure(observation) {
      clearProgressiveWaiting();
      clearTracePolling();
      const message = observation?.error?.message || "The model backend could not produce a Generated Outcome.";
      generationFailureMessage.textContent = message;
      systemStatus.textContent = "Observation failed. Retry or reset the sealed system.";
      observeButton.disabled = false;
      retryButton.disabled = false;
      setFailureActions(true);
      show(generationFailurePanel);
    }

    async function observe() {
      observeButton.disabled = true;
      retryButton.disabled = true;
      setFailureActions(false);
      clearTraceFrame();
      outcomeMetadata.textContent = "";
      revealNote.textContent = "";
      systemStatus.textContent = "Entropy sampled. Branch selected by the Model Backend.";
      show(waitingPanel);
      startProgressiveWaitingTimer();
      pollTrace();

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
      } finally {
        clearTracePolling();
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
      const outcomeLabel = observation.outcome === "living" ? "Living Cat" : "Dead Cat";
      const frameCount = observation.metadata.traceFrameCount || observation.traceRefs.length;
      outcomeMetadata.textContent = `Branch: ${outcomeLabel} | Seed: ${observation.metadata.seed} | Trace frames: ${frameCount}`;
      revealNote.textContent = observation.revealNote;
      systemStatus.textContent = "Final Generated Outcome received from the Model Backend.";
      generatedOutcome.onload = () => {
        generatedOutcome.dataset.loading = "false";
        clearProgressiveWaiting();
        show(revealedPanel);
        bringActivePanelIntoView(revealedPanel);
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
