# Generated Outcome File References

The first Model Backend will return a local file reference plus generation metadata for each Generated Outcome, rather than embedding image bytes or base64 data in the response. The model spike already writes local image files, generated outcomes are ephemeral in v1, and file references keep the backend contract small while the Browser UI and backend are both local.

**Considered Options**

- Return a local generated-image file reference plus metadata.
- Return raw image bytes.
- Return base64-encoded image data in JSON.

**Consequences**

The Browser UI will need a way to load the referenced local image from the Model Backend. The Model Backend remains responsible for writing each Generated Outcome and returning enough metadata for Dev Controls and debugging.
