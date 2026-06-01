## Problem Statement

Catbox has proven that a local diffusion model can produce both a Living-Cat Outcome and an Absent-Cat Outcome within the updated Primary Runtime Target, but the working path still lives in a throwaway model spike. A user cannot yet perform an Observation through Catbox and receive a real Generated Outcome through a stable Model Backend contract. Without this backend boundary, Browser UI work would either fake the core experience or couple directly to spike code that was not designed as a service.

## Solution

Build a thin persistent local Model Backend that preloads the `sd_turbo_img2img` pipeline once, accepts Observation requests, performs Outcome Selection for the normal experience, generates a real Ephemeral Outcome, writes the generated image to a local runtime path, and returns a file reference plus metadata. Add only the minimum Browser UI integration needed to prove that the Browser UI can observe, wait, reveal the Generated Outcome, handle Generation Failure, and reset. Keep visual polish, galleries, sharing, remote hosting, and fallback model work out of this PRD.

## User Stories

1. As a Catbox viewer, I want to open the sealed box, so that I can observe Catbox resolving into one outcome.
2. As a Catbox viewer, I want the observation to run a real local diffusion model, so that the reveal is not a fake static asset.
3. As a Catbox viewer, I want the box to resolve into either a Living-Cat Outcome or an Absent-Cat Outcome, so that the core Catbox premise is preserved.
4. As a Catbox viewer, I want the Generated Outcome to appear after waiting, so that the experience feels like the model produced it for this observation.
5. As a Catbox viewer, I want the Living-Cat Outcome to be immediately recognizable, so that I can tell the box contains a living cat.
6. As a Catbox viewer, I want the Absent-Cat Outcome to be immediately recognizable, so that I can tell the box is eerily empty.
7. As a Catbox viewer, I want both outcomes to preserve the Box Composition, so that they feel like alternate states of the same sealed box.
8. As a Catbox viewer, I want generation to complete within the Primary Runtime Target after the backend is ready, so that the wait feels acceptable.
9. As a Catbox viewer, I want Observation Noise while generation runs, so that the interface does not feel frozen.
10. As a Catbox viewer, I want Progressive Waiting if generation takes long enough, so that I know Catbox is still working.
11. As a Catbox viewer, I want a Reveal Note after the Generated Outcome appears, so that I understand that a local diffusion model created the image.
12. As a Catbox viewer, I want to reset after a Generated Outcome, so that I can observe Catbox again.
13. As a Catbox viewer, I want a clear Generation Failure state if generation fails, so that I am not shown a fake outcome.
14. As a Catbox viewer, I want retry from a Generation Failure state, so that a transient backend failure does not end the session.
15. As a Catbox viewer, I want reset from a Generation Failure state, so that I can return to the sealed box.
16. As a developer, I want the Model Backend to preload the model once at startup, so that observations do not pay the full model-load cost.
17. As a developer, I want the Model Backend to own Outcome Selection, so that the backend is the source of truth for the Generated Outcome.
18. As a developer, I want Dev Controls to force a specific outcome, so that I can tune and reproduce Living-Cat Outcome and Absent-Cat Outcome behavior.
19. As a developer, I want generated images written to local runtime paths, so that the Browser UI can load them without embedding large image payloads in JSON.
20. As a developer, I want generation metadata returned with each Generated Outcome, so that I can inspect outcome, seed, strength, steps, timing, and errors.
21. As a developer, I want structured backend readiness information, so that the Browser UI can distinguish startup from generation.
22. As a developer, I want the backend contract to separate normal Observation from Dev Controls overrides, so that production behavior and tuning behavior do not blur.
23. As a developer, I want backend errors to be structured, so that the Browser UI can present Generation Failure and Dev Controls can diagnose problems.
24. As a developer, I want generated outcomes to remain ephemeral, so that v1 avoids storage, gallery, and history scope.
25. As a developer, I want tests around the backend contract and outcome selection behavior, so that later UI work can rely on a stable interface.
26. As a developer, I want the spike generation logic moved or wrapped behind a deeper generation module, so that service code does not duplicate diffusion setup details.
27. As a developer, I want the Browser UI integration to use the same contract the real UI will use, so that the thin slice proves the product boundary.
28. As a developer, I want manual GPU validation documented, so that the local model path can be verified on the target machine.

## Implementation Decisions

- Build a persistent local Model Backend. It loads the preferred SD Turbo image-to-image pipeline once at service startup and remains available for observations.
- Observation latency is measured after the Model Backend is ready. Model startup can exceed the Primary Runtime Target.
- Use `sd_turbo_img2img` as the preferred generation path for this PRD. Do not continue fallback model investigation under the current WSL memory cap.
- The Model Backend owns Outcome Selection for the normal experience.
- Dev Controls may request a specific outcome for tuning and reproducibility, but the normal Browser UI does not choose the outcome.
- Generated Outcomes are returned as local file references plus metadata, not raw bytes or base64 JSON.
- Generated Outcomes are ephemeral runtime artifacts. Do not create history, gallery, save, or share behavior.
- Generation Failure is explicit. Do not substitute static images or fake generated fallbacks.
- The backend should expose a readiness operation so the Browser UI can know whether the model is loaded.
- The backend should expose an observation operation for normal use. The response should include selected outcome, generated image reference, timing metadata, seed/config metadata, and any Reveal Note data needed by the Browser UI.
- The backend should expose a development operation or development-only request fields that allow forced outcome and reproducible seed/config overrides.
- Extract a model runner module that owns pipeline loading, prompt configuration, Box Composition loading, generation, timing, and output writing behind a small interface.
- Extract an outcome selection module that owns ordinary-random selection and supports deterministic override for Dev Controls.
- Extract a response contract module or schema so Browser UI and Model Backend agree on success and Generation Failure shapes.
- Keep the Browser UI implementation minimal in this PRD: sealed box, observe action, waiting state, generated image reveal, failure state, and reset.
- Keep Browser UI polish secondary. Do not spend this PRD on animations, elaborate styling, sound, save/share, or a gallery.

## Testing Decisions

- Tests should verify external behavior and contracts rather than implementation details of Diffusers internals.
- Unit test the outcome selection module: normal selection returns only valid outcomes, forced Dev Controls outcomes are respected, invalid forced outcomes fail clearly.
- Unit test backend response contract validation: success responses include outcome, image reference, timing, and metadata; failure responses include structured error information.
- Unit test generation orchestration with a fake model runner: observation calls select an outcome, invoke generation once, and return the expected response shape.
- Unit test Generation Failure handling with a fake failing model runner: backend returns structured failure without a fake Generated Outcome.
- Unit test that generated output metadata represents Ephemeral Outcome behavior and does not require history persistence.
- Add an integration-style backend test using a fake or stubbed generator so the service contract can be tested without GPU access.
- Keep real SD Turbo GPU validation as a documented manual check because CI or sandboxed test runs may not have CUDA access.
- Prior art: the model spike metadata and persistent batch evidence define the timing and metadata fields that the first backend contract should preserve.

## Out of Scope

- Browser UI polish beyond the minimum observe, wait, reveal, failure, and reset flow.
- Static or fake generated fallbacks.
- Save, share, gallery, generated outcome history, or persistent user storage.
- Remote model hosting or cloud inference.
- Real quantum randomness.
- Authentication, multi-user sessions, deployment, or production hardening.
- Additional model fallback candidates while the current WSL memory cap remains unchanged.
- Upgrading PyTorch solely to unblock `.bin` model loading.
- Mobile-specific polish or accessibility refinement beyond basic usable controls.
- Sound design or advanced Observation Noise.

## Further Notes

- This PRD follows ADR-0001 through ADR-0005.
- The model spike passed with `sd_turbo_img2img` under the updated under-23-second Primary Runtime Target.
- The persistent batch evidence showed a recognizable Living-Cat Outcome at 20.323s generation after preload and a clean Absent-Cat Outcome at 7.722s generation after preload.
- GPU access was blocked inside the default sandbox during the spike; manual generation commands required escalated execution.
- The current known-working PyTorch setup is `torch==2.5.1` from the CUDA 12.1 wheel index.
