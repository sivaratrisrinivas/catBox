# Model Spike Summary

Status: passed

## Verdict

Passed for model feasibility.

The spike passes under the updated Primary Runtime Target of under 23 seconds. After cleaning the Box Composition and removing cat language from the Absent-Cat Outcome prompt, `sd_turbo_img2img` produced recognizable outputs for both Catbox outcomes on the preferred GPU path. A persistent preloaded SD Turbo process also produced both outcomes in one process, with the absent generation comfortably below the target and the recognizable living generation below the updated target.

## Candidate Results

| Candidate | Living-Cat Outcome | Absent-Cat Outcome | Runtime | Notes |
| --- | --- | --- | --- | --- |
| `sd_turbo_img2img` | Pass on cleaned Box Composition with seed `41100`, `--strength 0.8 --steps 4`; output `20260601T221414Z_sd_turbo_img2img_living_41100`. Persistent batch output `20260601T222511Z_sd_turbo_img2img_persistent_batch/01_living_41100.png` is also recognizable and recorded 20.323s generation | Pass on cleaned Box Composition with seed `41100`, `--strength 0.55 --steps 2`; output `20260601T221234Z_sd_turbo_img2img_absent_41100`. Persistent batch output `20260601T222511Z_sd_turbo_img2img_persistent_batch/02_absent_41100.png` is a clean empty box and recorded 7.722s generation after preload | One-shot recognizable living run recorded 21.701s generation; one-shot absent run recorded 20.216s generation. Persistent recognizable living run recorded 20.323s generation after a 31.487s model load. Persistent absent run recorded 7.722s generation. All passing SD Turbo evidence is below the updated under-23s Primary Runtime Target | Cleaning the source removed an animal-like oval, and the absent prompt now says empty/no animal without repeating cat. SD Turbo still has weak absent control and no negative-prompt support, but it is viable enough for the next Catbox phase |
| `sd15_conservative_img2img` | Not completed | Not completed | First run completed denoising but process exited 137 before metadata/image write | Download succeeded, but the process was OOM-killed under the current WSL memory limit after denoising |
| `tiny_sd_fallback_img2img` | Not run | Failed before generation | 23.867s failure before generation after initial loader attempt; 187.501s failure after `.bin` loading attempt | `segmind/tiny-sd` does not provide safetensors weights, and loading `.bin` weights is blocked by the installed `torch==2.5.1` security restriction requiring `torch>=2.6` |
| `bk_sdm_v2_tiny_img2img` | Not run | Not completed | 512px, 384px, and 256px attempts completed denoising but exited 137 before metadata/image write | Added as a safetensors fallback candidate using `nota-ai/bk-sdm-v2-tiny`, but it is still not practical under the current WSL memory cap |

## Decision

Do not spend more time on fallback model candidates under the current WSL memory cap.

Next best step: move from model feasibility into a thin Model Backend spike that preloads `sd_turbo_img2img`, exposes the two Catbox outcomes, and returns generated images to the Browser UI. Keep Browser UI polish secondary until the backend contract is working.
