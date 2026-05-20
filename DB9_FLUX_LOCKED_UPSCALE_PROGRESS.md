# DB9 Flux Locked Upscale - Progress Handoff

Date: 2026-05-20
Repo: https://github.com/bemapa3/db9_flux_locked_upscale

## Goal

Create a new independent ComfyUI node package for Flux upscale workflows. It must not modify or depend on the old DB9 node package.

The package is intended to solve these problems from the previous upscale nodes:

- wrong frame crop
- stretched aspect ratio
- visible seams between tiles
- color/tone/light/contrast drift from the source image
- LoRA upscale causing framing drift, likely because the LoRA was trained on mixed aspect ratios

## User Verified Baseline

The current known-good workflow:

- Model: Flux 2 Klein 9B
- KSampler denoise: 1.0
- CFG: 1.0
- This baseline does not drift frame by itself
- Drift appears when adding an upscale LoRA

VRAM context:

- Running on Colab/G4-like environment
- Approx VRAM mentioned: 95GB
- Best working size for Flux 2 Klein appears around 2048px

## New Package

Package name: `db9_flux_locked_upscale`

---

## v2 Rewrite — 2026-05-20 (commit 1508ad2)

### Bugs Fixed

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| Tile overlap/distortion (only top-left tile correct) | CPU/CUDA device mismatch — sampled tensor from VAE decode was on CUDA, canvas on CPU | Added `.to(cpu)` after every `_sample_tile()` call; canvas explicitly kept on CPU |
| Node runs forever with no output | No progress feedback | Added `comfy.utils.ProgressBar(total_tiles)` + per-tile console print |

### New Features

| Feature | Node | Details |
|---------|------|---------|
| Upscale model port | `DB9FluxLockedControlNetUpscale` | Optional `UPSCALE_MODEL` input — plugs in Real-ESRGAN etc for base upscale pass |
| More control guide types | `DB9GlobalControlGuide` | Added: `canny`, `tile_blur`, `line_art`, `depth_approx` |
| Real depth estimation | `DB9MiDaSDepth` (NEW) | Intel MiDaS via torch.hub — auto-downloads weights, cached in memory |
| HuggingFace ControlNet download | `DB9FluxControlNetLoader` (NEW) | 5 presets: Tile/Upscaler (jasperai), Union (InstantX), Canny/Depth/Line Art (XLabs v3) |
| Tile split/merge | `DB9TileSplit` + `DB9TileMerge` (NEW) | Split large image into tile batch → process independently → merge with cosine feather |

### All Registered Nodes (v2)

```
DB9GlobalControlGuide          — edge/depth guide image generator
DB9MiDaSDepth                  — real MiDaS depth map
DB9FluxControlNetLoader        — download Flux ControlNet from HuggingFace
DB9ColorToneLock               — post-process color/contrast correction
DB9TileSplit                   — split image into tile batch (N, tile_h, tile_w, C)
DB9TileMerge                   — merge tile batch back to full image
DB9FluxLockedControlNetUpscale — all-in-one tiled upscale (main node)
```

### Recommended Workflow for 6000-7000px Output

```
Input Image (e.g. 1500x1000px)
    |
    v
DB9TileSplit (tile=1536, overlap=192)
    |
    v (batch of N tiles, each 1536x1536)
KSampler / img2img with Flux 2 Klein 9B
    |
    v (N processed tiles)
DB9TileMerge (feather=128, reference_image=original for color lock)
    |
    v
Full output (e.g. 6000x4000px)
```

For all-in-one (smaller outputs up to ~4K):
```
Input Image
    |
    v
DB9FluxLockedControlNetUpscale (scale=2-4, tile=1536, overlap=192)
    |
    v
Upscaled image + base reference
```

### Notes for Next Session

- [ ] Test tile overlap fix on real Colab run
- [ ] Test DB9FluxControlNetLoader download with jasperai Tile preset
- [ ] Verify DB9MiDaSDepth works with MiDaS_small on T4/A100
- [ ] Consider adding `DB9TileControlPrep` — auto-generate tile control image per-tile during split
- [ ] Color lock global pass (after merge) may be too strong at default 0.55 * 0.75 — consider lowering
