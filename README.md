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
- `feather` — width of the cross-fade band at the middle of each tile overlap (default 128). Since v4.2 each pixel comes from one tile; only this band is blended.
- `color_lock_strength` — per-tile + global color correction (default 0.55)
- `contrast_lock_strength` — per-tile + global contrast correction (default 0.20)
- `upscale_model` *(optional)* — UPSCALE_MODEL for pixel-space base pass (Nomos8kSC, Real-ESRGAN, UltraSharp…). Works on ComfyUI ≥ 0.3x since v4.2 (was silently falling back to bicubic before).
- `control_net / control_image` *(optional)* — ControlNet guidance per tile
- `arch_detail_lock` *(optional, v4)* — keep the base image structure above ~7 px in the sampled tile, take only fine grain from Flux. Stops material/pattern hallucination on buildings (0 = off, 0.5 recommended).
- `veg_denoise` *(optional, v4)* — second sampling pass for foliage/grass at a higher denoise, blended by an automatic vegetation mask (0 = off, 0.40–0.45 recommended).
- `veg_mask_threshold` / `veg_mask_blur` *(optional, v4)* — vegetation mask tuning (excess-green threshold, softness in px).
- `positive_veg` *(optional, v4)* — separate prompt for the vegetation pass (e.g. "raw photo, photorealistic vegetation, individual leaves, natural grass blades").
- `veg_include_flowers` *(optional, v4.3)* — also send white/pink flowering shrubs to the vegetation pass.
- `texture_limit` *(optional, v4.3)* — in architecture areas, cap Flux detail energy at N × the base image's local detail (0 = off, 1.25 recommended). Stops invented cracks / mosaic on plain walls and sand.
- `vram_mode` *(optional, v4.3)* — `auto` (default) reads GPU memory: ≤16 GB → tile 1024 + tiled VAE; 24 GB → tile 1280 + tiled VAE; 32 GB → batch ≤2; 40–48 GB → batch ≤3; 80 GB+ → batch ≤6. Batch is then sized from the measured memory of the first tile; on OOM it halves the batch, then switches to tiled VAE.
- `tile_batch` *(optional, v4.3)* — force a batch size (0 = automatic).
- `veg_blur_boost` *(optional, v4.4, experimental)* — extra pass with more denoise for the blurriest vegetation interior. 0 = off (recommended): blending two differently drawn passes can leave translucent leaves; raise `veg_denoise` instead.
- `veg_blur_fraction` *(optional, v4.4)* — share of the vegetation (least sharp first) that gets the boost, default 0.35.

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
Post-process color and contrast correction against a reference image (global mean/std).

---

### DB9 Exact Color Restore *(v4.2, reworked in v4.3)*
Locks colours to the original image in CIE Lab. Only the low-frequency luminance and chroma are
corrected towards `reference`; since v4.3 the correction is guided-filtered by the new image so it
follows the new edges (no halos or ghost tints where Flux moved leaves). Put it last, before Save,
with `reference` = the original (un-upscaled) image.

**Inputs:** `image`, `reference`, `luma_radius` (default 32 px), `chroma_radius` (default 24 px),
`luma_strength`, `chroma_strength` (default 1.0), `edge_aware` (default on),
`tone_match` *(v4.4, default 1.0)* — after the spatial lock, match the mean a/b of every lightness
band (shadows → highlights) to the reference, measured at ~2k so sharpness differences don't bias
it. Fixes casts that cancel out on average (e.g. greener shadows while highlights are fine). 0 = off.
**Outputs:** `image`

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

### Tested preset: architecture + landscape, 2k → 6k (v4.4)

v4.3 preset with `veg_denoise` **0.48**, `veg_blur_boost` 0, Exact Color Restore `tone_match` 1.0,
then **ColorMatch** (ComfyUI-KJNodes, method `hm-mvgd-hm`, `image_ref` = original) before Save.
v4.4 also (no new inputs) matches the low frequencies of the vegetation pass to the main pass and
grows the vegetation mask into sky only — no more bright halos / translucent leaves at canopy edges.
Workflow: `workflows/DB9_FluxLockedUpscale_v4.4.0_2026-11-06_api.json`.

### Preset (v4.3)

Same as the v4.2 table below, plus: `veg_include_flowers` on, `texture_limit` 1.25,
`veg_mask_blur` 8, `vram_mode` auto, Exact Color Restore 32 / 24 / edge_aware, and remove
"crisp fine material texture" from the building prompt.

### Previous preset (v4.2)

| Parameter | Value |
|-----------|-------|
| upscale_model | 4xNomos8kSC |
| scale | 3.0 |
| tile_width / tile_height / overlap / feather | 1536 / 192 / 128 |
| steps / cfg / sampler / scheduler | 20 / 1.0 / euler / simple |
| denoise (buildings) | 0.28 |
| arch_detail_lock | 0.5 |
| veg_denoise (+ `positive_veg`) | 0.42 |
| sharpen_strength | 0.0 |
| color_lock / contrast_lock | 0.75 / 0.20 |
| Final node | DB9 Exact Color Restore (16 / 4 / 1.0 / 1.0) |

Avoid noise-offset LoRAs in this chain: they shift shadows and colour.