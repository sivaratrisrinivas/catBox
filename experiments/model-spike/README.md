# Catbox Model Spike

This is throwaway spike code for proving whether Catbox can generate a Recognizable Outcome locally before the Model Backend or Browser UI exist.

## Question

Can a local diffusion model transform one shared Box Composition into both:

- a Living-Cat Outcome
- an Absent-Cat Outcome

while preserving the box scene and plausibly meeting the Primary Runtime Target of under 23 seconds on the preferred GPU path?

## Run

Install dependencies:

```bash
uv sync
```

CUDA note: if `uv sync` installs a CPU-only PyTorch build, install the CUDA build that matches the machine before running GPU candidates. The target machine previously reported CUDA 12.2, so a CUDA 12.1 or compatible PyTorch build may be appropriate depending on current PyTorch support.

Create the handmade Box Composition only:

```bash
uv run python experiments/model-spike/generate.py --init-only
```

List candidates:

```bash
uv run python experiments/model-spike/generate.py --list
```

Run one candidate:

```bash
uv run python experiments/model-spike/generate.py --candidate sd_turbo_img2img --outcome all
```

Tune a cached candidate:

```bash
uv run python experiments/model-spike/generate.py --candidate sd_turbo_img2img --outcome living --strength 0.8 --steps 4
```

Measure a persistent preloaded SD Turbo process:

```bash
uv run python experiments/model-spike/generate.py --candidate sd_turbo_img2img --persistent-batch
```

Run all candidates:

```bash
uv run python experiments/model-spike/generate.py --candidate all --outcome all
```

## Candidates

The harness starts with three candidate shapes:

- `sd_turbo_img2img`: fast low-step image-to-image path using `stabilityai/sd-turbo`
- `sd15_conservative_img2img`: memory-conservative SD 1.5-style image-to-image path
- `tiny_sd_fallback_img2img`: smaller fallback candidate using `segmind/tiny-sd`

These are spike candidates, not product commitments. Replace them if current model availability or local CUDA behavior makes a candidate unsuitable.

## Evidence

Generated images and per-run metadata are written to `experiments/model-spike/outputs/`, which is gitignored.

Each metadata file records:

- candidate id
- model id
- device
- outcome
- seed
- prompt
- negative prompt
- strength
- steps
- resolution
- elapsed seconds
- model load seconds
- generation seconds
- output image path
- error details, if generation failed

The persistent batch writes one `metadata.json` containing a shared model-load time and per-generation timings for both outcomes.

After comparing runs, write the verdict in `experiments/model-spike/results/summary.md`.

## Pass/Fail Gate

The spike passes only if manual review finds at least:

- one Recognizable Outcome for the Living-Cat Outcome
- one Recognizable Outcome for the Absent-Cat Outcome
- enough Box Composition preservation that both feel like the same Catbox world
- a preferred GPU path that plausibly meets the under-23-second Primary Runtime Target

If no candidate passes, do not proceed to Browser UI polish. Revisit the model strategy first.

## Source Notes

- Hugging Face Diffusers documents `AutoPipelineForImage2Image` for selecting image-to-image pipelines from pretrained model ids.
- Diffusers Stable Diffusion img2img conditions generation on an initial image plus prompt.
- Diffusers memory documentation covers CPU offload and other strategies for limited GPU memory.
