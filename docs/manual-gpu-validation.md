# Manual GPU Validation for the First Observation Flow

Use this checklist on the target local GPU machine to validate the complete thin
Catbox flow. The goal is to prove that the persistent Model Backend reaches
readiness, both Recognizable Outcomes can be generated through Dev Controls, and
the normal Browser UI can observe and reveal a real Generated Outcome.

## Start the Model Backend and Browser UI

The current local product path starts the persistent Model Backend and Browser UI
in one process:

```bash
python -m catbox.browser_ui
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
- `outcome` is either `living` or `absent`.
- `imageRef` points at a real local PNG file.
- `metadata.generationSeconds` records generation timing after backend readiness.
- `metadata.ephemeral` is `true`.

Compare `metadata.generationSeconds` against the Primary Runtime Target: the
preferred GPU path should produce a Recognizable Outcome in under 23 seconds.
Record the observed time in the issue or validation notes.

## Force Both Outcomes Through Dev Controls

Use the development-only validation command to force both outcomes with a stable
seed:

```bash
python -m catbox.validate_sd_turbo_runner --outcome all --seed 41100
```

To validate only one outcome:

```bash
python -m catbox.validate_sd_turbo_runner --outcome living --seed 41100
python -m catbox.validate_sd_turbo_runner --outcome absent --seed 41100
```

Expected response:

- The first entry reports backend readiness.
- The Living-Cat Outcome entry has `status: "generated"` and `outcome:
  "living"`.
- The Absent-Cat Outcome entry has `status: "generated"` and `outcome:
  "absent"`.
- Each generated entry includes `metadata.generationSeconds`.

Compare each forced outcome's `metadata.generationSeconds` against the under 23
seconds Primary Runtime Target after readiness.

The current preferred SD Turbo settings are outcome-specific: the Living-Cat
Outcome should use 384px image-to-image generation with 4 steps, and the
Absent-Cat Outcome should use 512px image-to-image generation with 2 steps.

## Ephemeral Generated Outcomes

Generated files are written under:

```text
.runtime/generated-outcomes
```

Confirm that each `imageRef` points to a file in that runtime output area. These
are Ephemeral Outcomes for local display and debugging only. This validation
must confirm no gallery/history/save/share behavior was introduced.

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
python -m unittest tests.test_browser_ui.BrowserUiServerTests.test_generation_failure_returns_structured_failure_without_generated_image
```
