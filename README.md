# catBox

Catbox is an interactive image-generation demo where opening a sealed box triggers
a local diffusion model and resolves into either a Living-Cat Outcome or an
Absent-Cat Outcome.

## Thin Model Backend contract

The first implementation slice is a fake-backed Model Backend contract. It gives
Catbox a stable Python boundary before the real SD Turbo runner is wired in:

- `readiness()` reports whether the Model Backend is ready for observations.
- `observe()` performs backend-owned Outcome Selection for the normal experience.
- Successful observations return a Generated Outcome contract with an outcome,
  local image file reference, timing/config metadata, and a Reveal Note.
- Failed observations return an explicit Generation Failure contract and never
  substitute a static or fake Generated Outcome.

The fake runner is a test double, not the product experience. It lets the backend
contract, Outcome Selection, failure behavior, and Browser UI integration target
be tested quickly without CUDA, Diffusers, model downloads, or 20-second
generation runs. The real product path still belongs behind the same Model Runner
interface and must generate actual Ephemeral Outcomes.

## Why this exists

The model spike proved that `sd_turbo_img2img` can produce Recognizable Outcomes
within the Primary Runtime Target after the model is preloaded. This backend
contract turns that spike evidence into a product boundary:

- the Browser UI observes Catbox instead of choosing the outcome;
- the Model Backend is the source of truth for the selected Outcome;
- generated images are returned as local file references plus metadata;
- failures stay visible as Generation Failure rather than being hidden by static
  fallbacks.

## How to verify

Run the contract tests:

```bash
python -m unittest discover -s tests
```

These tests use the fake Model Runner and do not require GPU access. Manual GPU
validation is a separate step once the persistent SD Turbo runner is connected.
