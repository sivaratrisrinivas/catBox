# Persistent Model Backend

Catbox will use a persistent local Model Backend that loads the preferred diffusion pipeline once at service startup and stays running for observations. The model spike showed that one-shot CLI runs spend substantial time on model loading, while a preloaded SD Turbo path can produce both Recognizable Outcomes within the updated Primary Runtime Target. Catbox will not design the v1 observation flow around spawning a fresh model process per observation.

**Considered Options**

- Persistent local service with a preloaded pipeline.
- One-shot process per observation.

**Consequences**

The Browser UI should treat the Model Backend as a local service dependency and send observations to it after startup. Startup can take longer than an observation, but observation latency is measured against generation after the backend is ready.
