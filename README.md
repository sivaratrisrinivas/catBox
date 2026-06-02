# catBox

Catbox is an interactive image-generation demo. You open a sealed box, a local
image model runs, and the box turns into either a cozy cat scene or an eerie
empty-box scene.

## Model backend

The current code defines the small Python boundary that the browser will call
later:

- `readiness()` says whether the model backend is ready.
- `observe()` is the normal path. The backend chooses which scene to generate.
- `observe_with_dev_controls(...)` is for development only. It can force the cat
  scene or the empty-box scene, and it can reuse a seed or generation settings.
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

## Why this exists

The model spike showed that `sd_turbo_img2img` can create recognizable cat and
empty-box images after the model is loaded. This backend turns that experiment
into code the rest of Catbox can use:

- the browser asks to observe the box instead of choosing the result itself;
- the backend is the source of truth for which scene was chosen;
- Dev Controls can force `living` or `absent` without changing normal
  observation behavior;
- Dev Controls can pass a seed and generation settings to reproduce a run;
- generated images are returned as local file paths plus details about the run;
- failures stay visible instead of being hidden behind placeholder images.

## How to verify

Run the contract tests:

```bash
python -m unittest discover -s tests
```

These tests use fakes and stubs around the expensive model path. They do not
need GPU access or model downloads.

Manual GPU validation for the preferred local machine:

```bash
python -m catbox.validate_sd_turbo_runner --outcome all --seed 41100
```

That command preloads SD Turbo once, forces both Catbox outcomes through the
development-only path, writes ephemeral generated images under `.runtime/`, and
prints the same response shape the Browser UI will use.

The current preferred settings were manually validated on the local CUDA path:
the Living-Cat Outcome uses 384px image-to-image generation with 2 steps, and
the Absent-Cat Outcome uses 512px image-to-image generation with 2 steps.
