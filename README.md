# db9_flux_locked_upscale

ComfyUI custom node package for tiled upscale workflows with Flux 2 Klein 9B.
Solves: frame crop, aspect ratio stretch, tile seams, color/tone drift, LoRA framing drift.

**Independent** — does not modify or depend on any other DB9 package.

---

## Nodes

### DB9 Flux Locked ControlNet Upscale
All-in-one tiled upscale. Handles: base upscale → tile sampling → color lock → seam blending.

**Inputs:**
- `image` — source image
- `model`, `vae`, `positive`, `negative` — standard Flux workflow
- `scale` — output scale multiplier (e.g. 2.0 for 2x)
- `tile_width / tile_height` — tile size (default 1536, best for Flux 2 Klein)
- `overlap` — tile overlap in pixels (default 192)
- `feather` — cosine blend radius (default 128)
- `color_lock_strength` — per-tile + global color correction (default 0.55)
- `contrast_lock_strength` — per-tile + global contrast correction (default 0.20)
- `upscale_model` *(optional)* — UPSCALE_MODEL for pixel-space base pass (Real-ESRGAN etc)
- `control_net / control_image` *(optional)* — ControlNet guidance per tile

**Outputs:** `image` (upscaled), `base_reference` (pixel-upscaled base for comparison)

---

### DB9 Tile Split
Split a large image into overlapping tiles for independent processing.

**Inputs:** `image`, `tile_width`, `tile_height`, `overlap`
**Outputs:** `tiles` (IMAGE batch N×H×W×C), `tile_layout` (TILE_LAYOUT metadata)

---

### DB9 Tile Merge
Merge processed tiles back into full image with cosine feather blending.
Auto-resizes tiles if KSampler/VAE changed their dimensions.

**Inputs:** `tiles`, `tile_layout`, `feather`, `reference_image` *(optional)*, `color_lock_strength` *(optional)*
**Outputs:** `image`

**Large image workflow (6000-7000px+):**
```
Input → DB9TileSplit → KSampler → DB9TileMerge → Full output
```

---

### DB9 MiDaS Depth
Real monocular depth estimation using Intel MiDaS. Weights auto-downloaded via torch.hub.

**Inputs:** `image`, `model_type` (MiDaS_small / DPT_Hybrid / DPT_Large), `detect_resolution`, `invert`
**Outputs:** `depth_map`

---

### DB9 Flux ControlNet Loader
Download Flux-compatible ControlNet from HuggingFace and load as CONTROL_NET.
Saves to ComfyUI's `models/controlnet/` directory.

**Presets:**
- Tile/Upscaler — jasperai/Flux.1-dev-Controlnet-Upscaler *(recommended for upscale)*
- Union ControlNet — InstantX/FLUX.1-dev-Controlnet-Union *(Canny/Depth/Pose/Tile)*
- Canny / Depth / Line Art — XLabs-AI v3

**Inputs:** `preset`, `hf_token` (blank for public models), `force_redownload`
**Outputs:** `control_net`

---

### DB9 Global Control Guide
Generate edge/depth control images without external preprocessors.

**Guide types:** `sobel`, `soft_edge`, `canny`, `tile_blur`, `line_art`, `depth_approx`

---

### DB9 Color Tone Lock
Post-process color and contrast correction against a reference image.

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/bemapa3/db9_flux_locked_upscale
```

Optional dependencies:
```bash
pip install huggingface_hub   # for DB9FluxControlNetLoader
pip install Pillow            # for DB9MiDaSDepth (PIL resize)
```

MiDaS model weights are downloaded automatically on first use via `torch.hub`.

---

## Recommended Settings for Flux 2 Klein 9B

| Parameter | Value |
|-----------|-------|
| tile_width / tile_height | 1536 |
| overlap | 192 |
| feather | 128 |
| denoise | 1.0 |
| cfg | 1.0 |
| color_lock_strength | 0.45 - 0.65 |
| contrast_lock_strength | 0.15 - 0.25 |