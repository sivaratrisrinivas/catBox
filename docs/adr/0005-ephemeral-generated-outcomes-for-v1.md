# Ephemeral Generated Outcomes for v1

Catbox v1 will treat Generated Outcomes as ephemeral runtime artifacts, not as saved history, a gallery, or shareable user content. Generated image files may exist locally long enough for display and debugging, but the product will not promise long-term storage or retrieval in v1.

**Considered Options**

- Ephemeral local generated outcomes only.
- Persist a generation history.
- Build save, gallery, or share behavior.

**Consequences**

The Model Backend can write images to local runtime output paths without introducing a storage model. Browser UI work should focus on observation, reveal, failure, and reset rather than history management.
