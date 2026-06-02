from __future__ import annotations

import argparse
import json
from pathlib import Path

from catbox.env_loader import DEFAULT_ENV_FILE, load_env_file
from catbox.model_backend import CatboxModelBackend, Outcome
from catbox.sd_turbo_runner import DEFAULT_RUNTIME_DIR, SdTurboImageToImageModelRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manually validate Catbox's persistent SD Turbo Model Runner."
    )
    parser.add_argument("--seed", type=int, default=41100)
    parser.add_argument("--outcome", choices=["living", "absent", "all"], default="all")
    parser.add_argument("--runtime-dir", default=str(DEFAULT_RUNTIME_DIR))
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    runner = SdTurboImageToImageModelRunner(runtime_dir=Path(args.runtime_dir))
    backend = CatboxModelBackend(model_runner=runner)

    responses: list[dict[str, object]] = [{"readiness": backend.readiness()}]
    outcomes: list[Outcome] = ["living", "absent"] if args.outcome == "all" else [args.outcome]
    for outcome in outcomes:
        responses.append(
            backend.observe_with_dev_controls(
                {
                    "outcome": outcome,
                    "seed": args.seed,
                }
            )
        )

    print(json.dumps(responses, indent=2))


if __name__ == "__main__":
    main()
