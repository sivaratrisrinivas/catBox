---
title: Catbox
sdk: docker
app_port: 7860
---

# catBox

Catbox is an interactive image-generation demo. You open a sealed box, a local
image model runs, and the box turns into either a cozy cat scene or an eerie
lifeless-cat scene.

## What this project is

Catbox is a thin local product slice around a real image model. The user-facing
experience is intentionally small: the Browser UI starts from a sealed box, the
user observes it, and the local Model Backend generates one real Generated
Outcome. The outcome is either a Living-Cat Outcome or a Dead-Cat Outcome.

The project is not a static reveal, a canvas-only animation, a gallery, or a
prompt playground. The normal path does not let the Browser UI choose the
outcome or tune generation. Those controls exist only in development paths for
validation and reproducibility.

## Why this exists

The model spike showed that `sd_turbo_img2img` can create recognizable cat and
empty-box images after the model is loaded; Catbox now uses that same path for a
living-cat branch and a non-graphic dead-cat branch. This backend turns that
experiment into code the rest of Catbox can use:

- the browser asks to observe the box instead of choosing the result itself;
- the backend is the source of truth for which scene was chosen;
- Dev Controls can force `living` or `dead` without changing normal
  observation behavior;
- Dev Controls can pass a seed and generation settings to reproduce a run;
- generated images are returned as local file paths plus details about the run;
- real intermediate frames can be captured as a single-branch Captured
  Denoising Trace when the runner supports it;
- failures stay visible instead of being hidden behind placeholder images.

Catbox v1 treats generated files as Ephemeral Outcomes. They may exist locally
long enough for display and debugging, but the product does not provide
gallery/history/save/share behavior.

## Model backend and Browser UI

The current code defines the small Python boundary that the Browser UI calls:

- `readiness()` says whether the model backend is ready.
- `observe()` is the normal path. The backend chooses which scene to generate.
- `observe_with_dev_controls(...)` is for development only. It can force the
  living-cat scene or the dead-cat scene, and it can reuse a seed or generation
  settings.
- Successful observations return the chosen scene, a local image file path,
  timing details, and a short note for the reveal.
- Failed observations return a clear error instead of pretending that an image
  was generated.

The SD Turbo runner is the real product path for the first Model Backend. It
loads `stabilityai/sd-turbo` once, loads the shared Box Composition, picks the
right Outcome Prompt, writes an ephemeral generated image under `.runtime/`, and
returns the local file path plus generation metadata through the backend
contract.

The fake runner is only for tests and early wiring. It lets the backend behavior
be checked quickly without CUDA, model downloads, or slow image generation.

The local Browser UI is served by `python -m catbox.browser_ui`. It is a
full-screen sealed-system interface rather than a card-based demo. It sends a
normal observation request without choosing an outcome, polls for real Captured
Denoising Trace frames while the Model Backend runs, and reveals the generated
image only after the backend-provided local file reference has loaded.
Observation Noise remains a fallback before the first captured frame appears,
and the trace surface is intentionally treated as a secondary instrument view
instead of competing with the final Generated Outcome. The Browser UI keeps
each state in normal document flow so the header, apparatus view, status text,
and controls do not overlap. When the generated outcome loads, the page scrolls
the revealed apparatus into view so the image is visible inside the viewport,
with the final image presented as the dominant visual state. The Browser UI
keeps the Model Backend authoritative, shows the Reveal Note, and supports Reset
back to the sealed box. If generation fails, the Browser UI shows Generation
Failure with Retry and Reset instead of registering or serving a fake generated
image.

## How the project fits together

- `catbox/model_backend.py` owns readiness, Outcome Selection, Dev Controls, and
  the observation response contract.
- `catbox/sd_turbo_runner.py` owns the persistent SD Turbo image-to-image runner,
  prompt selection, generation settings, timing, and ephemeral file output.
- `catbox/browser_ui.py` owns the local Browser UI, readiness polling, normal
  observation requests, trace polling, generated image serving, Progressive
  Waiting, Generation Failure, Retry, and Reset.
- `catbox/validate_sd_turbo_runner.py` is the manual Dev Controls validation
  entrypoint for forcing outcomes and running Outcome Visibility tuning on the
  target GPU runtime.
- `tests/` covers the public backend, runner, Browser UI, and manual validation
  documentation contracts with fakes and stubs instead of requiring GPU access.
- `docs/adr/` records the project decisions that keep the Browser UI thin, the
  Model Backend authoritative, and Generated Outcomes ephemeral.

## How to verify

Run the contract tests:

```bash
uv run python -m unittest discover -s tests
```

These tests use fakes and stubs around the expensive model path. They do not
need GPU access or model downloads.

If you want Hugging Face authenticated downloads, create a local `.env` file:

```bash
cp .env.example .env
```

Then edit `.env`:

```text
HF_TOKEN=hf_your_read_token_here
```

The Browser UI startup command and manual validation command load `.env` before
the SD Turbo runner requests model files. `.env` is ignored by git and should
not be committed.

Run the local Browser UI:

```bash
uv run python -m catbox.browser_ui
```

Then open `http://127.0.0.1:8765`. The page starts from the sealed box, sends a
normal observation request to the Model Backend without choosing an outcome, and
keeps backend startup separate from active observation. During observation it
polls the trace endpoint and displays Captured Denoising Trace frames as they
are registered by the Model Backend, including a final-trace display from the
completed observation response if polling misses the last frame. It reveals a
subtle Progressive Waiting status only if generation takes long enough, reveals
the generated image from the returned local file reference, scrolls the revealed
apparatus into view, shows the Reveal Note, and lets Reset return to the sealed
box. The layout is vertically scrollable on small screens, and inactive states
are removed from layout so they cannot cover the active state or controls. If
generation fails, the Browser UI shows a Generation Failure state with Retry and
Reset instead of substituting a static image or fake Generated Outcome. The
first real run may download model files before the observation completes.

Manual GPU validation for the preferred GPU runtime:

```bash
uv run python -m catbox.validate_sd_turbo_runner --outcome all --seed 41100
```

That command preloads SD Turbo once, forces both Catbox outcomes through the
development-only path, writes ephemeral generated images under `.runtime/`, and
prints the same response shape the Browser UI will use.

To test one explicit tuning candidate, pass generation settings through Dev
Controls:

```bash
uv run python -m catbox.validate_sd_turbo_runner --outcome dead --seed 41100 --steps 6 --strength 0.7 --width 512 --height 512
```

To run the first Outcome Visibility matrix on the deployed GPU runtime:

```bash
uv run python -m catbox.validate_sd_turbo_runner --matrix --seed 41100
```

The default matrix tries both outcomes with steps `4,6,8`, sizes `512,768`,
Living-Cat Outcome strengths `0.75,0.8,0.85`, and Dead-Cat Outcome strengths
`0.6,0.7,0.8`. Choose the fastest passing candidate where both final Generated
Outcomes are immediately recognizable, the Dead-Cat Outcome remains
non-graphic, the shared Box Composition is still legible, and
`metadata.generationSeconds` stays under the Primary Runtime Target.

The current tuned SD Turbo defaults emphasize Outcome Visibility: both outcomes
use 512px image-to-image generation with 6 steps. The Living-Cat Outcome uses
strength `0.78`, and the Dead-Cat Outcome uses strength `0.7`.

For the complete first-observation GPU validation checklist, including Browser
UI readiness, normal observation, forced Dev Controls outcomes, runtime timing,
ephemeral output files, and failure retry/reset behavior, see
`docs/manual-gpu-validation.md`.

## Deploy on Hugging Face Spaces

The preferred public deployment target is a Docker Hugging Face Space with GPU
hardware. The container serves Catbox on port `7860`, which is the default
external port for Spaces.

Create a new Hugging Face Space with these settings:

- SDK: Docker
- Visibility: Public or Protected
- Hardware: start with `1x Nvidia L4`; use `Nvidia A10G - small` if L4 is too
  slow or unavailable

Add a Space secret named `HF_TOKEN` if the model download requires
authenticated Hugging Face access.

Push this repository's contents to the Space git remote. The Space README
metadata at the top of this file tells Hugging Face to build the Docker image
and expose port `7860`. The Space will start `python -m catbox.browser_ui`,
download `sd-turbo` on first startup if it is not already cached, and serve the
app from the Space URL.

Generated outcomes and Captured Denoising Trace frames are still ephemeral in
the deployed container. They are written under `.runtime/generated-outcomes` and
may be lost when the Space restarts.
