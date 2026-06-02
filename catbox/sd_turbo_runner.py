from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from catbox.model_backend import Outcome


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BOX_COMPOSITION = ROOT / "experiments" / "model-spike" / "assets" / "box-composition.png"
DEFAULT_RUNTIME_DIR = ROOT / ".runtime" / "generated-outcomes"


@dataclass(frozen=True)
class SdTurboImageToImageConfig:
    id: str = "sd_turbo_img2img"
    model_id: str = "stabilityai/sd-turbo"
    steps: int = 2
    strength: float = 0.55
    guidance_scale: float = 0.0
    width: int = 512
    height: int = 512
    variant: str | None = "fp16"
    use_safetensors: bool = True
    supports_negative_prompt: bool = False


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

OUTCOME_DEFAULTS: dict[Outcome, dict[str, object]] = {
    "living": {"steps": 4, "strength": 0.8, "width": 384, "height": 384},
    "absent": {"steps": 2, "strength": 0.55},
}

GENERATION_CONFIG_FIELDS = {"steps", "strength", "guidance_scale", "width", "height"}


class SdTurboImageToImageModelRunner:
    def __init__(
        self,
        runtime_dir: str | Path = DEFAULT_RUNTIME_DIR,
        box_composition_path: str | Path = DEFAULT_BOX_COMPOSITION,
        config: SdTurboImageToImageConfig | None = None,
        pipeline_loader: Callable[..., object] | None = None,
        box_image_loader: Callable[[Path, SdTurboImageToImageConfig], object] | None = None,
        torch_module: object | None = None,
        now: Callable[[], str] | None = None,
        timer: Callable[[], float] | None = None,
    ) -> None:
        self._runtime_dir = Path(runtime_dir)
        self._box_composition_path = Path(box_composition_path)
        self._config = config or SdTurboImageToImageConfig()
        self._pipeline_loader = pipeline_loader or self._default_pipeline_loader
        self._box_image_loader = box_image_loader or self._default_box_image_loader
        self._torch = torch_module
        self._now = now or (lambda: datetime.now(timezone.utc).isoformat())
        self._timer = timer or time.perf_counter
        self._device: str | None = None
        self._pipeline: object | None = None
        self._load_error: Exception | None = None

        try:
            if self._torch is None:
                self._torch = self._load_torch()
            self._device = self._select_device()
            self._ensure_box_composition()
            self._pipeline = self._load_pipeline()
        except Exception as error:
            self._load_error = error

    def is_ready(self) -> bool:
        return self._pipeline is not None and self._load_error is None

    def generate(
        self,
        outcome: Outcome,
        seed: int,
        config: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if not self.is_ready():
            message = "SD Turbo image-to-image runner is not ready."
            if self._load_error is not None:
                message = f"{message} {self._load_error}"
            raise RuntimeError(message)
        if self._torch is None or self._device is None:
            raise RuntimeError("SD Turbo image-to-image runner is not ready.")

        run_config = self._config_for(outcome, config)
        init_image = self._box_image_loader(self._box_composition_path, run_config)
        generator_device = self._device if self._device == "cuda" else "cpu"
        generator = self._torch.Generator(device=generator_device).manual_seed(seed)

        call_kwargs: dict[str, object] = {
            "prompt": PROMPTS[outcome],
            "image": init_image,
            "strength": run_config.strength,
            "num_inference_steps": run_config.steps,
            "guidance_scale": run_config.guidance_scale,
            "generator": generator,
        }
        if run_config.supports_negative_prompt:
            call_kwargs["negative_prompt"] = NEGATIVE_PROMPTS[outcome]

        started_at = self._now()
        generation_start = self._timer()
        result = self._pipeline(**call_kwargs)
        generation_seconds = round(self._timer() - generation_start, 3)

        self._runtime_dir.mkdir(parents=True, exist_ok=True)
        image_path = self._runtime_dir / self._output_filename(started_at, outcome, seed)
        result.images[0].save(image_path)

        return {
            "image_ref": str(image_path),
            "generation_seconds": generation_seconds,
            "metadata": {
                "runner": run_config.id,
                "modelId": run_config.model_id,
                "device": self._device,
                "boxComposition": str(self._box_composition_path),
                "prompt": PROMPTS[outcome],
                "negativePrompt": (
                    NEGATIVE_PROMPTS[outcome] if run_config.supports_negative_prompt else None
                ),
                "config": asdict(run_config),
            },
        }

    def _load_pipeline(self) -> object:
        if self._torch is None or self._device is None:
            raise RuntimeError("SD Turbo dependencies are not loaded.")

        dtype = self._torch.float16 if self._device == "cuda" else self._torch.float32
        kwargs: dict[str, object] = {
            "torch_dtype": dtype,
            "use_safetensors": self._config.use_safetensors,
        }
        if self._config.variant and self._device == "cuda":
            kwargs["variant"] = self._config.variant

        pipe = self._pipeline_loader(self._config.model_id, **kwargs)
        pipe = pipe.to(self._device)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        return pipe

    def _config_for(
        self, outcome: Outcome, config_overrides: Mapping[str, object] | None
    ) -> SdTurboImageToImageConfig:
        values = dict(OUTCOME_DEFAULTS[outcome])
        if config_overrides:
            values.update(
                {
                    key: value
                    for key, value in config_overrides.items()
                    if key in GENERATION_CONFIG_FIELDS
                }
            )
        return replace(self._config, **values)

    def _ensure_box_composition(self) -> None:
        if self._box_composition_path.exists():
            return

        from PIL import Image, ImageDraw

        self._box_composition_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (512, 512), (238, 232, 218))
        draw = ImageDraw.Draw(image)
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
        image.save(self._box_composition_path)

    def _select_device(self) -> str:
        if self._torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def _output_filename(started_at: str, outcome: Outcome, seed: int) -> str:
        safe_started_at = (
            started_at.replace(":", "")
            .replace("-", "")
            .replace(".", "")
            .replace("+", "Z")
        )
        return f"{safe_started_at}_{outcome}_{seed}.png"

    @staticmethod
    def _load_torch() -> object:
        import torch

        return torch

    @staticmethod
    def _default_pipeline_loader(model_id: str, **kwargs: object) -> object:
        from diffusers import AutoPipelineForImage2Image

        return AutoPipelineForImage2Image.from_pretrained(model_id, **kwargs)

    @staticmethod
    def _default_box_image_loader(
        box_composition_path: Path, config: SdTurboImageToImageConfig
    ) -> object:
        from PIL import Image

        return (
            Image.open(box_composition_path)
            .convert("RGB")
            .resize((config.width, config.height))
        )
