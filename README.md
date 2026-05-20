# DB9 Flux Locked Upscale

Independent ComfyUI node package for Flux tiled upscale with structure lock, ControlNet per tile, seam blending, and color/tone matching.

## Nodes

### DB9 Global Control Guide
Creates a small global structure guide from the source image.

### DB9 Color Tone Lock
Matches color, tone, and contrast back toward the reference image.

### DB9 Flux Locked ControlNet Upscale
Tiled upscale wrapper with optional ControlNet per tile.

## Recommended starting settings

- scale: 2.0
- tile_width: 1536 or 2048
- tile_height: 1536 or 2048
- overlap: 192 to 256
- feather: 128 to 160
- denoise: 1.0
- cfg: 1.0

## Why this package exists

It is designed for cases where an upscale LoRA changes framing or causes ratio drift.  
This package keeps the aspect ratio, applies a global structural guide, uses tile overlap blending, and adds color/tone locking to preserve the source look.