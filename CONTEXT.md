# Catbox

Catbox is an interactive image-generation demo where a user clicks a sealed box and watches a local diffusion model resolve it into a cozy cat or eerie absence.

## Language

**Catbox**:
An interactive diffusion demo where opening a sealed box triggers a real model run that resolves into either a living-cat outcome or an eerie absent-cat outcome.
_Avoid_: Canvas-only toy, static art reveal, fake diffusion demo

**Outcome**:
One of the final observable states produced by Catbox after the box is opened. The two planned outcomes are the living-cat outcome and the absent-cat outcome.
_Avoid_: Ending, result, branch result

**Absent-Cat Outcome**:
An outcome showing an empty box with symbolic signs of absence, such as a collar, toy, or faint impression. It should be eerie without depicting gore, explicit death, or harm.
_Avoid_: Dead-cat outcome, ghost-cat outcome, gore branch, cartoon death

**Living-Cat Outcome**:
An outcome showing a cozy living cat inside the same box composition used by the absent-cat outcome.
_Avoid_: Cat portrait, action reveal, mascot image

**Outcome Selection**:
The ordinary-random choice made at observation time that commits Catbox to one outcome.
_Avoid_: Quantum collapse, predetermined reveal, hidden user choice

**Generated Outcome**:
The final image produced by the diffusion model for the selected outcome.
_Avoid_: Static asset, reveal image, fake output

**Ephemeral Outcome**:
A generated outcome that exists only for the current Catbox session and is not automatically saved by the product.
_Avoid_: Saved image, collectible, generation history

**Recognizable Outcome**:
A generated outcome that a viewer can immediately identify as either a cat in a box or an eerie empty box, even if the image is low-resolution or visually imperfect.
_Avoid_: Polished output, perfect image, abstract result

**Box Composition**:
The shared source image or sketch that anchors every Catbox generation around the same box scene before it becomes a living-cat outcome or absent-cat outcome.
_Avoid_: Input image, starter image, base asset

**Observation**:
The user's click or tap that opens the sealed box and commits Catbox to generating one outcome.
_Avoid_: Click, open action, measurement

**Reset**:
The user's action after a generated outcome appears that returns Catbox to the sealed box for another observation.
_Avoid_: Refresh, reroll, replay

**Dev Controls**:
Hidden controls for tuning or reproducing Catbox generation, such as prompt, seed, model path, and outcome probability.
_Avoid_: User controls, playground controls, settings

**Outcome Prompt**:
The prompt template that guides the model toward either the living-cat outcome or the absent-cat outcome while preserving the box composition.
_Avoid_: User prompt, random prompt, prompt playground

**Observation Noise**:
The theatrical visual and audio transition shown after observation while Catbox waits for the generated outcome. It suggests diffusion without claiming to show the model's internal denoising steps.
_Avoid_: Real denoising, model trace, intermediate frames

**Progressive Waiting**:
The waiting behavior where Catbox begins with Observation Noise, then reveals subtle status only if generation takes long enough that the user might think it is stuck.
_Avoid_: Loading screen, debug panel, immediate spinner

**Generation Failure**:
A user-visible state shown when the Model Backend cannot produce a Generated Outcome for the current observation. It should offer retry or reset without pretending that a real outcome was generated.
_Avoid_: Fake fallback, static reveal, hidden failure

**Primary Runtime Target**:
The expectation that the preferred GPU generation path should produce a recognizable generated outcome in under 23 seconds.
_Avoid_: Instant generation, unlimited wait, benchmark goal

**Reveal Note**:
The short plain-language explanation shown after the generated outcome appears. It connects the experience to diffusion without interrupting the observation flow.
_Avoid_: Tutorial, lecture, debug explanation

**Model Backend**:
The local Python service that runs Catbox's diffusion model and returns generated images to the browser UI.
_Avoid_: API server, inference server, backend

**Model Fallback Ladder**:
The ordered set of local generation paths Catbox may use when the preferred GPU model path is unavailable or too slow. Each path must still produce a real Generated Outcome.
_Avoid_: Fake fallback, cached reveal, remote escape hatch

**Browser UI**:
The local web interface where the user observes Catbox and sees the generated outcome.
_Avoid_: Frontend, webpage, client

## Example Dialogue

**Developer**: When does the model start running?

**Domain expert**: The model starts after observation, when the user opens Catbox.

**Developer**: Is Catbox allowed to fake the diffusion process with a canvas blend?

**Domain expert**: Not for the core experience. Catbox should run a small, fast diffusion model, even if the model is constrained.

**Developer**: Where does generation happen?

**Domain expert**: Generation happens in the Model Backend. The Browser UI is responsible for interaction and presentation.

**Developer**: What happens if the preferred GPU model path does not work?

**Domain expert**: Catbox uses the Model Fallback Ladder, but every accepted path must still create a real Generated Outcome.

**Developer**: Does the visual noise have to be real model denoising?

**Domain expert**: No. In v1, Observation Noise is presentation while the Model Backend creates a real Generated Outcome.

**Developer**: What if generation is slow?

**Domain expert**: Catbox uses Progressive Waiting: preserve the theatrical reveal first, then show subtle status if needed.

**Developer**: What if generation fails?

**Domain expert**: Catbox shows a Generation Failure state with retry or reset. It should not fake a Generated Outcome.

**Developer**: How fast should the primary model path be?

**Domain expert**: The Primary Runtime Target is under 23 seconds for a Recognizable Outcome.

**Developer**: When should Catbox explain diffusion?

**Domain expert**: Catbox should show a Reveal Note after the Generated Outcome appears, with optional technical details kept out of the main flow.

**Developer**: How good does the generated image need to be?

**Domain expert**: It needs to be a Recognizable Outcome. Catbox should prefer clear identity over polished image quality in v1.

**Developer**: Does Catbox preserve generated images?

**Domain expert**: Not in v1. Catbox treats each generated image as an Ephemeral Outcome.

**Developer**: How does the user observe again?

**Domain expert**: The Generated Outcome remains visible until the user chooses Reset, which returns Catbox to the sealed box.

**Developer**: Can users tune Catbox generation?

**Domain expert**: Not in the normal experience. Tuning belongs in Dev Controls, while users only observe and reset.

**Developer**: Are prompts part of the public experience?

**Domain expert**: No. Outcome Prompts have fixed product meaning, with only development-time style and tuning changes.

**Developer**: Should Catbox generate from a blank text prompt?

**Domain expert**: No. Catbox should transform a shared Box Composition so both outcomes still feel like they came from the same sealed box.

**Developer**: Is the outcome truly quantum?

**Domain expert**: No. Outcome Selection is quantum-inspired ordinary randomness unless Catbox is explicitly connected to a real quantum randomness source.

**Developer**: Does v1 include real quantum randomness?

**Domain expert**: No. Real quantum randomness is outside v1; Outcome Selection uses ordinary randomness.

**Developer**: What does the absent branch show?

**Domain expert**: The Absent-Cat Outcome shows symbolic absence, not explicit death or harm.

**Developer**: What does the living branch show?

**Domain expert**: The Living-Cat Outcome shows a cozy cat inside the same Box Composition.
