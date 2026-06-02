# Manual GPU Validation for the First Observation Flow

Use this checklist on the target local GPU machine to validate the complete thin
Catbox flow. The goal is to prove that the persistent Model Backend reaches
readiness, both Recognizable Outcomes can be generated through Dev Controls, and
the normal Browser UI can observe and reveal a real Generated Outcome.

## Start the Model Backend and Browser UI

The current local product path starts the persistent Model Backend and Browser UI
in one process:

```bash
uv run python -m catbox.browser_ui
```

Open the Browser UI at:

```text
http://127.0.0.1:8765
```

If you want to verify readiness from a second terminal before observing:

```bash
curl http://127.0.0.1:8765/api/readiness
```

Wait for:

```json
{"status": "ready", "modelBackend": "ready"}
```

GPU execution may require escalated local commands, driver access, CUDA device
access, or model cache writes. Do not require `HF_TOKEN` for this validation
unless the user explicitly provides one.

## Run a Normal Observation

In the Browser UI, click Observe after readiness. The normal Browser UI path
must not choose the outcome; it sends the normal observation request to the Model
Backend.

The same public request can be run from a second terminal:

```bash
curl -X POST http://127.0.0.1:8765/api/observe
```

Expected response:

- `status` is `generated`.
- `outcome` is either `living` or `dead`.
- `imageRef` points at a real local PNG file.
- `traceRefs` contains zero or more real Captured Denoising Trace frame paths.
- `metadata.traceFrameCount` matches the number of returned trace frames when
  trace capture is supported by the runner.
- `metadata.generationSeconds` records generation timing after backend readiness.
- `metadata.ephemeral` is `true`.

Compare `metadata.generationSeconds` against the Primary Runtime Target: the
preferred GPU path should produce a Recognizable Outcome in under 23 seconds.
Record the observed time in the issue or validation notes.

While the Browser UI is observing, the trace endpoint can be checked from a
second terminal:

```bash
curl http://127.0.0.1:8765/api/trace
```

When SD Turbo trace capture is working, the response should show the selected
branch's captured frame paths as they are registered. It should not show a fake
dual-branch trace.

## Force Both Outcomes Through Dev Controls

Use the development-only validation command to force both outcomes with a stable
seed:

```bash
uv run python -m catbox.validate_sd_turbo_runner --outcome all --seed 41100
```

To validate only one outcome:

```bash
uv run python -m catbox.validate_sd_turbo_runner --outcome living --seed 41100
uv run python -m catbox.validate_sd_turbo_runner --outcome dead --seed 41100
```

To validate one explicit tuning candidate:

```bash
uv run python -m catbox.validate_sd_turbo_runner --outcome dead --seed 41100 --steps 6 --strength 0.7 --width 512 --height 512
```

To run the first Outcome Visibility matrix on the deployed GPU runtime:

```bash
uv run python -m catbox.validate_sd_turbo_runner --matrix --seed 41100
```

The default matrix tries both outcomes with steps `4,6,8`, sizes `512,768`,
Living-Cat Outcome strengths `0.75,0.8,0.85`, and Dead-Cat Outcome strengths
`0.6,0.7,0.8`. These values can be narrowed for a smaller run:

```bash
uv run python -m catbox.validate_sd_turbo_runner --matrix --seed 41100 --matrix-steps 4,6 --matrix-sizes 512 --living-strengths 0.75 --dead-strengths 0.6,0.7
```

Expected response:

- The first entry reports backend readiness.
- The Living-Cat Outcome entry has `status: "generated"` and `outcome:
  "living"`.
- The Dead-Cat Outcome entry has `status: "generated"` and `outcome:
  "dead"`.
- Each generated entry includes `metadata.generationSeconds`.

Compare each forced outcome's `metadata.generationSeconds` against the under 23
seconds Primary Runtime Target after readiness.

A matrix candidate passes when the final Living-Cat Outcome is immediately
recognizable, the final Dead-Cat Outcome is immediately recognizable and
non-graphic, both still feel anchored to the shared Box Composition, and
`metadata.generationSeconds` stays under the Primary Runtime Target. Prefer the
fastest passing candidate; if timings are close, choose the candidate with
clearer outcome identity.

The current preferred SD Turbo settings are outcome-specific and may be changed
through Dev Controls during validation. Outcome Visibility uses one shared
standard for both outcomes, but each outcome may use different tuning values.

## Ephemeral Generated Outcomes

Generated files are written under:

```text
.runtime/generated-outcomes
```

Confirm that each `imageRef` points to a file in that runtime output area. These
are Ephemeral Outcomes for local display and debugging only. This validation
must confirm no gallery/history/save/share behavior was introduced.

Captured Denoising Trace frames are also Ephemeral Outcomes and may be written
under a trace subdirectory in the runtime output area.

## Generation Failure, Retry, and Reset

If the Model Backend cannot produce a Generated Outcome, the Browser UI should
show Generation Failure instead of substituting a static image or fake generated
file.

Expected failure behavior:

- The observation response has `status: "generation_failed"`.
- The response does not include `imageRef`.
- The Browser UI shows the failure message.
- Retry runs the normal `POST /api/observe` path again.
- Reset clears the failure state and returns to the sealed box.

Exercise the retry/reset behavior by using the Browser UI after a real local
generation failure, such as unavailable model weights, CUDA out of memory, or an
unready runner. The browser-facing failure contract can also be checked without
GPU access:

```bash
uv run python -m unittest tests.test_browser_ui.BrowserUiServerTests.test_generation_failure_returns_structured_failure_without_generated_image
```
