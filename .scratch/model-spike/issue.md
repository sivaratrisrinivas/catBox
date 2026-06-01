# Model Spike: Prove Local Catbox Generation Feasibility

Labels: `ready-for-agent`

## Problem

Catbox's core promise depends on a local diffusion model producing a Recognizable Outcome from a shared Box Composition. Before building the Model Backend or Browser UI, we need evidence that the user's local hardware can generate both a Living-Cat Outcome and an Absent-Cat Outcome fast enough to support the experience.

The known target machine has an NVIDIA GeForce GTX 1050 Ti with 4 GB VRAM. The main uncertainty is whether a small, fast local model path can create recognizable image-to-image outputs under the Primary Runtime Target.

## Goal

Build a small model spike under `experiments/model-spike/` that compares 2-3 local generation paths using the same Box Composition and Outcome Prompts.

The spike should answer:

- Can a local model generate a recognizable cozy cat in the box?
- Can a local model generate a recognizable eerie absent-cat box without gore or explicit death?
- Can the preferred GPU path plausibly produce each outcome in under 20 seconds?
- Which candidate should become the first Model Backend generation path?

## Scope

Create a minimal spike harness, not product code.

The spike should include:

- A simple handmade Box Composition for testing.
- A script that runs candidate generation paths against the same Box Composition.
- Outcome Prompt templates for the Living-Cat Outcome and Absent-Cat Outcome.
- Output images for manual review.
- Metadata JSON for each run.
- A short summary of candidate results.

Raw output batches should be ignored by git. A tiny curated evidence set may be committed later only if useful.

## Candidate Shape

Compare 2-3 local generation paths:

- Fast low-step image-to-image path.
- Memory-conservative SD 1.5-style image-to-image path.
- CPU-compatible or smaller fallback path.

Choose exact model names during the spike after checking current official `diffusers` support, model availability, and local CUDA/PyTorch compatibility.

## Spike Structure

Use:

- `experiments/model-spike/README.md`
- `experiments/model-spike/generate.py`
- `experiments/model-spike/assets/box-composition.png`
- `experiments/model-spike/outputs/`
- `experiments/model-spike/results/`

Use `uv` and `pyproject.toml` for dependency management unless PyTorch/CUDA installation forces a documented exception.

## Metadata

Each run should record:

- Candidate id/name.
- Device used.
- Outcome requested.
- Seed.
- Prompt.
- Negative prompt, if used.
- Image-to-image strength.
- Step count.
- Resolution.
- Elapsed generation time.
- Whether the run used the primary GPU path or a fallback path.
- Error details, if generation failed.

## Success Criteria

The spike passes only if manual review finds at least:

- One Recognizable Outcome for the Living-Cat Outcome.
- One Recognizable Outcome for the Absent-Cat Outcome.
- Both preserve the Box Composition enough to feel like the same sealed-box world.
- The preferred GPU path is plausibly under the 20-second Primary Runtime Target.

If no candidate satisfies this, do not proceed to Browser UI polish. Revisit model strategy first.

## Out of Scope

- Browser UI implementation.
- FastAPI Model Backend implementation.
- Real quantum randomness.
- Real intermediate denoising visualization.
- User prompt editor.
- Product save/gallery/history.
- Remote hosted generation.
- Gore or explicit death imagery.
- Training or fine-tuning a custom model.
- Mobile-first polish.

## Follow-Up After Spike

If the spike passes:

- Write the first ADR for the chosen local generation strategy.
- Build the FastAPI + Pydantic Model Backend around the selected path.
- Use a job-based observe API.
- Add readiness reporting and single-active-job behavior.
- Then build the Vite + React + TypeScript Browser UI.

If the spike fails:

- Document which candidates failed and why.
- Decide whether to relax the runtime target, use a different model family, or revisit local-only scope.
