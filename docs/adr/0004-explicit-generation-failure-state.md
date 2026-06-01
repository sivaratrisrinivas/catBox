# Explicit Generation Failure State

When the Model Backend cannot produce a Generated Outcome, Catbox will show an explicit Generation Failure state with retry or reset instead of substituting a static placeholder or fake generated fallback. The core Catbox promise is that outcomes come from a real model run, so failures should be visible rather than disguised.

**Considered Options**

- Show an explicit failure state with retry or reset.
- Substitute a static fallback image.
- Pretend a generated fallback was produced.

**Consequences**

The Browser UI needs a failure state in the observation flow. The Model Backend should return structured error information when generation fails, and Dev Controls can expose enough metadata to diagnose failures.
