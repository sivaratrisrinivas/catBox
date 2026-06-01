# Model Backend Owns Outcome Selection

For the normal Catbox experience, the Model Backend will own Outcome Selection: the Browser UI sends an observation request, and the backend chooses whether to generate the Living-Cat Outcome or the Absent-Cat Outcome. Dev Controls may request a specific outcome for tuning and reproducibility, but that path is not part of the ordinary user experience.

**Considered Options**

- Browser UI chooses the outcome and asks the backend to generate it.
- Model Backend chooses the outcome during observation.
- Dev Controls can override the outcome only for development.

**Consequences**

The Browser UI remains responsible for observation, waiting, reveal, and reset presentation. The Model Backend becomes the source of truth for which Generated Outcome was produced, including the selected outcome and generation metadata returned with the image.
