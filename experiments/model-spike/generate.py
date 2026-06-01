from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"
OUTPUTS_DIR = ROOT / "outputs"
BOX_COMPOSITION = ASSETS_DIR / "box-composition.png"

Outcome = Literal["living", "absent"]


@dataclass(frozen=True)
class Candidate:
    id: str
    model_id: str
    description: str
    steps: int
    strength: float
    guidance_scale: float
    width: int
    height: int
    variant: str | None = None
    prefer_gpu: bool = True
    cpu_offload: bool = False
    supports_negative_prompt: bool = True
    use_safetensors: bool = True


CANDIDATES: dict[str, Candidate] = {
    "sd_turbo_img2img": Candidate(
        id="sd_turbo_img2img",
        model_id="stabilityai/sd-turbo",
        description="Fast low-step image-to-image candidate.",
        steps=2,
        strength=0.55,
        guidance_scale=0.0,
        width=512,
        height=512,
        variant="fp16",
        supports_negative_prompt=False,
    ),
    "sd15_conservative_img2img": Candidate(
        id="sd15_conservative_img2img",
        model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        description="Memory-conservative SD 1.5-style image-to-image candidate.",
        steps=15,
        strength=0.62,
        guidance_scale=6.5,
        width=256,
        height=256,
        variant="fp16",
        cpu_offload=True,
    ),
    "tiny_sd_fallback_img2img": Candidate(
        id="tiny_sd_fallback_img2img",
        model_id="segmind/tiny-sd",
        description="Smaller fallback image-to-image candidate.",
        steps=12,
        strength=0.68,
        guidance_scale=6.0,
        width=512,
        height=512,
        prefer_gpu=False,
        use_safetensors=False,
    ),
    "bk_sdm_v2_tiny_img2img": Candidate(
        id="bk_sdm_v2_tiny_img2img",
        model_id="nota-ai/bk-sdm-v2-tiny",
        description="Safetensors lightweight Stable Diffusion fallback candidate.",
        steps=12,
        strength=0.62,
        guidance_scale=6.5,
        width=512,
        height=512,
        variant="fp16",
        cpu_offload=True,
    ),
}


PROMPTS: dict[Outcome, str] = {
    "living": (
        "cozy living cat curled inside the same open cardboard box, warm soft light, "
        "gentle illustrated realism, clear cat in box, charming but not cartoonish"
    ),
    "absent": (
        "same open cardboard box but completely empty, small red collar and toy mouse inside, "
        "vacant interior, no animal present, eerie gentle atmosphere, no gore"
    ),
}

SD_TURBO_PERSISTENT_BATCH: tuple[tuple[Outcome, int, float, int], ...] = (
    ("living", 41100, 0.8, 4),
    ("absent", 41100, 0.55, 2),
)

BASE_NEGATIVE_PROMPT = (
    "gore, blood, injury, graphic death, horror violence, distorted box, extra boxes, "
    "text, watermark, blurry, low detail"
)

NEGATIVE_PROMPTS: dict[Outcome, str] = {
    "living": BASE_NEGATIVE_PROMPT,
    "absent": (
        f"{BASE_NEGATIVE_PROMPT}, cat, kitten, animal, pet, fur, ears, tail, whiskers, eyes, face"
    ),
}


def ensure_dirs() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def create_box_composition(path: Path = BOX_COMPOSITION) -> None:
    ensure_dirs()
    image = Image.new("RGB", (512, 512), (238, 232, 218))
    draw = ImageDraw.Draw(image)

    # Simple handmade source image: intentionally bland, stable, and easy to replace.
    draw.rectangle((0, 360, 512, 512), fill=(184, 169, 142))
    draw.polygon([(116, 230), (396, 230), (352, 386), (160, 386)], fill=(161, 117, 69))
    draw.polygon([(116, 230), (160, 386), (96, 342), (70, 208)], fill=(138, 93, 51))
    draw.polygon([(396, 230), (352, 386), (420, 342), (442, 208)], fill=(126, 82, 45))
    draw.polygon([(116, 230), (256, 150), (396, 230), (256, 286)], fill=(190, 140, 82))
    draw.polygon([(70, 208), (256, 150), (116, 230)], fill=(214, 164, 98))
    draw.polygon([(442, 208), (256, 150), (396, 230)], fill=(199, 147, 86))
    draw.line((256, 150, 256, 286), fill=(116, 74, 38), width=3)
    draw.line((116, 230, 396, 230), fill=(116, 74, 38), width=3)
    draw.polygon([(164, 330), (348, 330), (326, 374), (184, 374)], fill=(145, 107, 66))
    draw.text((174, 426), "CATBOX SOURCE", fill=(88, 74, 62))

    image.save(path)


def load_init_image(candidate: Candidate) -> Image.Image:
    if not BOX_COMPOSITION.exists():
        create_box_composition()
    return Image.open(BOX_COMPOSITION).convert("RGB").resize((candidate.width, candidate.height))


def select_device(candidate: Candidate, torch: Any) -> str:
    if candidate.prefer_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_pipeline(candidate: Candidate, device: str, torch: Any) -> Any:
    from diffusers import AutoPipelineForImage2Image

    dtype = torch.float16 if device == "cuda" else torch.float32
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "use_safetensors": candidate.use_safetensors,
    }
    if candidate.variant and device == "cuda":
        kwargs["variant"] = candidate.variant

    pipe = AutoPipelineForImage2Image.from_pretrained(candidate.model_id, **kwargs)

    if candidate.cpu_offload and device == "cuda" and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    return pipe


def generate_image(
    pipe: Any,
    candidate: Candidate,
    outcome: Outcome,
    seed: int,
    device: str,
    torch: Any,
) -> tuple[Image.Image, float]:
    init_image = load_init_image(candidate)
    generator_device = device if device == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)

    call_kwargs: dict[str, Any] = {
        "prompt": PROMPTS[outcome],
        "image": init_image,
        "strength": candidate.strength,
        "num_inference_steps": candidate.steps,
        "guidance_scale": candidate.guidance_scale,
        "generator": generator,
    }
    if candidate.supports_negative_prompt:
        call_kwargs["negative_prompt"] = NEGATIVE_PROMPTS[outcome]

    generation_start = time.perf_counter()
    image = pipe(**call_kwargs).images[0]
    return image, round(time.perf_counter() - generation_start, 3)


def base_metadata(candidate: Candidate, outcome: Outcome, seed: int, device: str) -> dict[str, Any]:
    return {
        "startedAt": datetime.now(timezone.utc).isoformat(),
        "candidate": asdict(candidate),
        "outcome": outcome,
        "seed": seed,
        "prompt": PROMPTS[outcome],
        "negativePrompt": NEGATIVE_PROMPTS[outcome] if candidate.supports_negative_prompt else None,
        "device": device,
        "boxComposition": str(BOX_COMPOSITION),
        "outputImage": None,
        "elapsedSeconds": None,
        "modelLoadSeconds": None,
        "generationSeconds": None,
        "error": None,
    }


def run_candidate(candidate: Candidate, outcome: Outcome, seed: int) -> dict[str, Any]:
    import torch

    ensure_dirs()
    device = select_device(candidate, torch)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{candidate.id}_{outcome}_{seed}"
    run_dir = OUTPUTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = base_metadata(candidate, outcome, seed, device)

    start = time.perf_counter()
    try:
        load_start = time.perf_counter()
        pipe = load_pipeline(candidate, device, torch)
        metadata["modelLoadSeconds"] = round(time.perf_counter() - load_start, 3)

        image, metadata["generationSeconds"] = generate_image(
            pipe, candidate, outcome, seed, device, torch
        )
        image_path = run_dir / "image.png"
        image.save(image_path)
        metadata["outputImage"] = str(image_path)
    except Exception as error:  # The spike records failures as evidence.
        metadata["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    finally:
        metadata["elapsedSeconds"] = round(time.perf_counter() - start, 3)
        metadata_path = run_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata


def run_persistent_batch(candidate: Candidate) -> dict[str, Any]:
    import torch

    ensure_dirs()
    device = select_device(candidate, torch)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{candidate.id}_persistent_batch"
    run_dir = OUTPUTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    batch_metadata: dict[str, Any] = {
        "startedAt": datetime.now(timezone.utc).isoformat(),
        "candidateId": candidate.id,
        "modelId": candidate.model_id,
        "device": device,
        "boxComposition": str(BOX_COMPOSITION),
        "modelLoadSeconds": None,
        "elapsedSeconds": None,
        "runs": [],
        "error": None,
    }

    start = time.perf_counter()
    try:
        load_start = time.perf_counter()
        pipe = load_pipeline(candidate, device, torch)
        batch_metadata["modelLoadSeconds"] = round(time.perf_counter() - load_start, 3)

        for index, (outcome, seed, strength, steps) in enumerate(SD_TURBO_PERSISTENT_BATCH, start=1):
            run_candidate_config = replace(candidate, strength=strength, steps=steps)
            run_metadata = base_metadata(run_candidate_config, outcome, seed, device)
            run_metadata["modelLoadSeconds"] = 0
            run_metadata["batchIndex"] = index

            run_start = time.perf_counter()
            try:
                image, run_metadata["generationSeconds"] = generate_image(
                    pipe, run_candidate_config, outcome, seed, device, torch
                )
                image_path = run_dir / f"{index:02d}_{outcome}_{seed}.png"
                image.save(image_path)
                run_metadata["outputImage"] = str(image_path)
            except Exception as error:  # Keep later batch evidence if one run fails.
                run_metadata["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
            finally:
                run_metadata["elapsedSeconds"] = round(time.perf_counter() - run_start, 3)
                batch_metadata["runs"].append(run_metadata)
    except Exception as error:
        batch_metadata["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    finally:
        batch_metadata["elapsedSeconds"] = round(time.perf_counter() - start, 3)
        metadata_path = run_dir / "metadata.json"
        metadata_path.write_text(json.dumps(batch_metadata, indent=2), encoding="utf-8")

    return batch_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Catbox local diffusion model spike candidates.")
    parser.add_argument("--candidate", default="all", help="Candidate id, or 'all'.")
    parser.add_argument("--outcome", choices=["living", "absent", "all"], default="all")
    parser.add_argument("--seed", type=int, default=41099)
    parser.add_argument("--steps", type=int, help="Override candidate inference step count.")
    parser.add_argument("--strength", type=float, help="Override candidate image-to-image strength.")
    parser.add_argument("--guidance-scale", type=float, help="Override candidate guidance scale.")
    parser.add_argument(
        "--persistent-batch",
        action="store_true",
        help="Load SD Turbo once and run the known living and absent settings in one process.",
    )
    parser.add_argument("--init-only", action="store_true", help="Only create the handmade Box Composition.")
    parser.add_argument("--list", action="store_true", help="List candidates and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    if args.init_only:
        create_box_composition()
        print(f"Created {BOX_COMPOSITION}")
        return

    if args.list:
        for candidate in CANDIDATES.values():
            print(f"{candidate.id}: {candidate.model_id} - {candidate.description}")
        return

    candidate_ids = list(CANDIDATES) if args.candidate == "all" else [args.candidate]
    unknown = [candidate_id for candidate_id in candidate_ids if candidate_id not in CANDIDATES]
    if unknown:
        raise SystemExit(f"Unknown candidate(s): {', '.join(unknown)}")

    if args.persistent_batch:
        if candidate_ids != ["sd_turbo_img2img"]:
            raise SystemExit("--persistent-batch currently supports only --candidate sd_turbo_img2img")
        if args.outcome != "all":
            raise SystemExit("--persistent-batch runs both outcomes; leave --outcome as all")
        metadata = run_persistent_batch(CANDIDATES["sd_turbo_img2img"])
        if metadata["error"]:
            print(f"failed: persistent batch -> {metadata['error']['message']}")
            return
        print(
            f"ok: persistent batch total={metadata['elapsedSeconds']}s "
            f"model_load={metadata['modelLoadSeconds']}s"
        )
        for run in metadata["runs"]:
            status = "failed" if run["error"] else "ok"
            print(
                f"{status}: {run['outcome']} strength={run['candidate']['strength']} "
                f"steps={run['candidate']['steps']} generation={run['generationSeconds']}s "
                f"-> {run['outputImage'] or run['error']['message']}"
            )
        return

    outcomes: list[Outcome] = ["living", "absent"] if args.outcome == "all" else [args.outcome]
    for candidate_id in candidate_ids:
        candidate = CANDIDATES[candidate_id]
        overrides: dict[str, Any] = {}
        if args.steps is not None:
            overrides["steps"] = args.steps
        if args.strength is not None:
            overrides["strength"] = args.strength
        if args.guidance_scale is not None:
            overrides["guidance_scale"] = args.guidance_scale
        if overrides:
            candidate = replace(candidate, **overrides)

        for outcome in outcomes:
            metadata = run_candidate(candidate, outcome, args.seed)
            status = "failed" if metadata["error"] else "ok"
            print(
                f"{status}: {candidate_id}/{outcome} total={metadata['elapsedSeconds']}s "
                f"generation={metadata['generationSeconds']}s "
                f"-> {metadata['outputImage'] or metadata['error']['message']}"
            )


if __name__ == "__main__":
    main()
