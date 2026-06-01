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

The fake runner is only for tests and early wiring. It lets the backend behavior
be checked quickly without CUDA, model downloads, or slow image generation. The
real product path still needs to run a real local image model behind the same
runner interface.

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

These tests use the fake runner and do not need GPU access. Manual GPU testing
comes later, once the real SD Turbo runner is connected.
