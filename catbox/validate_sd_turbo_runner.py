from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from catbox.env_loader import DEFAULT_ENV_FILE, load_env_file
from catbox.model_backend import CatboxModelBackend, Outcome
from catbox.sd_turbo_runner import DEFAULT_RUNTIME_DIR, SdTurboImageToImageModelRunner

DEFAULT_MATRIX_STEPS = "4,6,8"
DEFAULT_MATRIX_SIZES = "512,768"
DEFAULT_LIVING_STRENGTHS = "0.75,0.8,0.85"
DEFAULT_DEAD_STRENGTHS = "0.6,0.7,0.8"


@dataclass(frozen=True)
class ValidationRun:
    outcome: Outcome
    seed: int
    config: dict[str, object] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manually validate Catbox's persistent SD Turbo Model Runner."
    )
    parser.add_argument("--seed", type=int, default=41100)
    parser.add_argument("--outcome", choices=["living", "dead", "all"], default="all")
    parser.add_argument("--runtime-dir", default=str(DEFAULT_RUNTIME_DIR))
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--strength", type=float)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Run an Outcome Visibility tuning matrix for the selected outcomes.",
    )
    parser.add_argument("--matrix-steps", default=DEFAULT_MATRIX_STEPS)
    parser.add_argument("--matrix-sizes", default=DEFAULT_MATRIX_SIZES)
    parser.add_argument("--living-strengths", default=DEFAULT_LIVING_STRENGTHS)
    parser.add_argument("--dead-strengths", default=DEFAULT_DEAD_STRENGTHS)
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def config_overrides_from_args(args: argparse.Namespace) -> dict[str, object] | None:
    overrides = {
        key: value
        for key, value in {
            "steps": args.steps,
            "strength": args.strength,
            "width": args.width,
            "height": args.height,
        }.items()
        if value is not None
    }
    if not overrides:
        return None
    return overrides


def selected_outcomes(outcome: str) -> list[Outcome]:
    return ["living", "dead"] if outcome == "all" else [outcome]


def validation_runs_from_args(args: argparse.Namespace) -> list[ValidationRun]:
    outcomes = selected_outcomes(args.outcome)
    if not args.matrix:
        return [
            ValidationRun(
                outcome=outcome,
                seed=args.seed,
                config=config_overrides_from_args(args),
            )
            for outcome in outcomes
        ]

    steps_values = parse_int_list(args.matrix_steps)
    size_values = parse_int_list(args.matrix_sizes)
    strengths_by_outcome = {
        "living": parse_float_list(args.living_strengths),
        "dead": parse_float_list(args.dead_strengths),
    }
    runs: list[ValidationRun] = []
    for outcome in outcomes:
        for steps in steps_values:
            for strength in strengths_by_outcome[outcome]:
                for size in size_values:
                    runs.append(
                        ValidationRun(
                            outcome=outcome,
                            seed=args.seed,
                            config={
                                "steps": steps,
                                "strength": strength,
                                "width": size,
                                "height": size,
                            },
                        )
                    )
    return runs


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    runner = SdTurboImageToImageModelRunner(runtime_dir=Path(args.runtime_dir))
    backend = CatboxModelBackend(model_runner=runner)

    responses: list[dict[str, object]] = [{"readiness": backend.readiness()}]
    for run in validation_runs_from_args(args):
        controls: dict[str, object] = {
            "outcome": run.outcome,
            "seed": run.seed,
        }
        if run.config is not None:
            controls["config"] = run.config
        responses.append(
            backend.observe_with_dev_controls(controls)
        )

    print(json.dumps(responses, indent=2))


if __name__ == "__main__":
    main()
