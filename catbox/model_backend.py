from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Literal, Protocol, TypedDict

Outcome = Literal["living", "absent"]

VALID_OUTCOMES: set[str] = {"living", "absent"}

_ONE_PIXEL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02"
    b"\xfeA\x89\x81\x0b\x00\x00\x00\x00IEND\xaeB`\x82"
)


class GeneratedOutcome(TypedDict):
    status: Literal["generated"]
    outcome: Outcome
    imageRef: str
    metadata: dict[str, object]
    revealNote: str


class GenerationFailure(TypedDict):
    status: Literal["generation_failed"]
    error: dict[str, str]
    metadata: dict[str, object]


ObservationResponse = GeneratedOutcome | GenerationFailure


class ModelRunner(Protocol):
    def is_ready(self) -> bool: ...

    def generate(self, outcome: Outcome, seed: int) -> dict[str, object]: ...


class FakeModelRunner:
    def __init__(self, output_dir: str | Path, ready: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self._ready = ready
        self.generations: list[dict[str, object]] = []

    def is_ready(self) -> bool:
        return self._ready

    def generate(self, outcome: Outcome, seed: int) -> dict[str, object]:
        self.generations.append({"outcome": outcome, "seed": seed})
        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_path = self.output_dir / f"{outcome}_{seed}.png"
        image_path.write_bytes(_ONE_PIXEL_PNG)
        return {
            "image_ref": str(image_path),
            "generation_seconds": 0.001,
        }


class CatboxModelBackend:
    def __init__(
        self,
        model_runner: ModelRunner,
        seed_source: Callable[[], int] | None = None,
        outcome_source: Callable[[], str] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._model_runner = model_runner
        self._seed_source = seed_source or (lambda: random.SystemRandom().randint(1, 2**31 - 1))
        self._outcome_source = outcome_source or self._select_outcome
        self._clock = clock or (lambda: 0.0)

    def readiness(self) -> dict[str, str]:
        if self._model_runner.is_ready():
            return {"status": "ready", "modelBackend": "ready"}
        return {"status": "starting", "modelBackend": "starting"}

    def observe(self) -> ObservationResponse:
        seed = self._seed_source()
        outcome = self._outcome_source()
        if outcome not in VALID_OUTCOMES:
            return self._generation_failure(
                seed=seed,
                error_type="InvalidOutcome",
                message=f"Unsupported outcome: {outcome}",
            )

        started_at = self._clock()
        try:
            generated = self._model_runner.generate(outcome, seed)
        except Exception as error:
            return self._generation_failure(
                seed=seed,
                error_type=type(error).__name__,
                message=str(error),
                outcome=outcome,
            )

        return {
            "status": "generated",
            "outcome": outcome,
            "imageRef": str(generated["image_ref"]),
            "metadata": {
                "seed": seed,
                "startedAt": started_at,
                "generationSeconds": generated["generation_seconds"],
                "ephemeral": True,
            },
            "revealNote": "A local diffusion model generated this outcome for this observation.",
        }

    def _generation_failure(
        self,
        seed: int,
        error_type: str,
        message: str,
        outcome: Outcome | None = None,
    ) -> GenerationFailure:
        metadata: dict[str, object] = {
            "seed": seed,
            "ephemeral": True,
        }
        if outcome is not None:
            metadata["outcome"] = outcome
        return {
            "status": "generation_failed",
            "error": {
                "type": error_type,
                "message": message,
            },
            "metadata": metadata,
        }

    @staticmethod
    def _select_outcome() -> Outcome:
        return random.SystemRandom().choice(("living", "absent"))
