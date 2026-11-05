"""
db9_flux_locked_upscale / nodes.py
v4.2.0 — 2026-11-05 (see CHANGELOG.md)
  - upscale_model works on ComfyUI >= 0.3x (comfy_extras V3 ImageUpscaleWithModel)
  - vegetation pass (veg_denoise / positive_veg) + arch_detail_lock
  - new node DB9ExactColorRestore (Lab colour lock to the original)
  - center-cut seam blending (no more blurry bands along tile overlaps)

Rewrite v3 — 2026-06-01

v3 Fixes (detail blur on small structures):
   - Removed destructive double color/contrast lock (was applied per-tile AND global)
   - Added detail-preserving color lock: extracts HF detail before correction, re-adds after
   - Added unsharp mask post-merge for sharpness recovery
   - Canvas kept on CPU to avoid VRAM OOM on large outputs
   - Progress bar via comfy.utils.ProgressBar
   - Per-tile print so user can see progress in console

New nodes:
  - DB9MiDaSDepth          — real depth via MiDaS (torch.hub)
  - DB9FluxControlNetLoader — download + load Flux ControlNets from HuggingFace
  - DB9TileSplit            — split image into tile batch
  - DB9TileMerge            — merge tile batch back to full image

Updated nodes:
  - DB9GlobalControlGuide  — +canny, +tile_blur, +line_art, +depth_approx
  - DB9FluxLockedControlNetUpscale — +upscale_model port, +device fix, +progress
"""

import copy
import math
import os
import pathlib

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional ComfyUI imports (safe fallbacks for unit-testing outside ComfyUI)
# ---------------------------------------------------------------------------
try:
    import nodes as comfy_nodes
except Exception:
    comfy_nodes = None

try:
    import comfy.utils as _comfy_utils
except Exception:
    _comfy_utils = None

try:
    import comfy.controlnet as _comfy_cn
except Exception:
    _comfy_cn = None

try:
    import folder_paths as _folder_paths
except Exception:
    _folder_paths = None

try:
    from huggingface_hub import hf_hub_download as _hf_download
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

# MiDaS model cache (loaded once, reused across calls)
_MIDAS_CACHE = {"model": None, "transform": None, "model_type": None, "device": None}


# ---------------------------------------------------------------------------
# Helpers: path resolution
# ---------------------------------------------------------------------------

def _controlnet_dir() -> str:
    """Return ComfyUI's models/controlnet directory, creating it if needed."""
    if _folder_paths is not None:
        try:
            dirs = _folder_paths.get_folder_paths("controlnet")
            if dirs:
                return dirs[0]
        except Exception:
            pass
    # Walk up from this file to find ComfyUI root
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "models" / "controlnet"
        if candidate.exists():
            return str(candidate)
    # Fallback: ~/.cache/comfyui/controlnet
    fallback = pathlib.Path.home() / ".cache" / "comfyui" / "controlnet"
    fallback.mkdir(parents=True, exist_ok=True)
    return str(fallback)


# ---------------------------------------------------------------------------
# Tensor utilities
# ---------------------------------------------------------------------------

def _bhwc_to_nchw(image: torch.Tensor) -> torch.Tensor:
    """ComfyUI IMAGE (B,H,W,C) -> float32 (B,C,H,W)."""
    if image.ndim != 4:
        raise ValueError(f"Expected 4-D BHWC tensor, got shape {image.shape}")
    return image.movedim(-1, 1).float()


def _nchw_to_bhwc(image: torch.Tensor) -> torch.Tensor:
    """(B,C,H,W) -> ComfyUI IMAGE (B,H,W,C), clamped to [0,1]."""
    return image.movedim(1, -1).clamp(0.0, 1.0)


def _resize_nchw(x: torch.Tensor, size, mode: str = "bicubic") -> torch.Tensor:
    if mode == "nearest":
        return F.interpolate(x, size=size, mode=mode)
    if mode == "lanczos":
        mode = "bicubic"
    return F.interpolate(x, size=size, mode=mode, align_corners=False)


def _make_positions(size: int, tile: int, overlap: int):
    """
    Compute tile start positions along one axis so that:
      - Every pixel is covered by at least one tile.
      - Adjacent tiles overlap by `overlap` pixels.
      - The last tile always ends exactly at `size`.
    """
    if size <= tile:
        return [0]
    step = max(1, tile - overlap)
    positions = list(range(0, size - tile + 1, step))
    last = size - tile
    if positions[-1] != last:
        positions.append(last)
    return positions



def _seam_axis(positions, i, tile, size, feather, dtype=torch.float32):
    """1-D weights for tile i: 1 inside its own region, cosine cross-fade of width `feather`
    centred on the middle of each overlap. Neighbouring ramps are complementary (sum = 1)."""
    p = positions[i]
    n = min(tile, size - p)
    t = torch.arange(n, dtype=dtype) + (p + 0.5)
    w = torch.ones(n, dtype=dtype)
    if i > 0:
        o0, o1 = p, min(positions[i - 1] + tile, size)
        if o1 > o0:
            mid = 0.5 * (o0 + o1)
            f = float(max(1.0, min(float(feather), float(o1 - o0))))
            r = ((t - (mid - f / 2.0)) / f).clamp(0.0, 1.0)
            w = w * (0.5 - 0.5 * torch.cos(r * math.pi))
    if i < len(positions) - 1:
        o0, o1 = positions[i + 1], min(p + tile, size)
        if o1 > o0:
            mid = 0.5 * (o0 + o1)
            f = float(max(1.0, min(float(feather), float(o1 - o0))))
            r = (((mid + f / 2.0) - t) / f).clamp(0.0, 1.0)
            w = w * (0.5 - 0.5 * torch.cos(r * math.pi))
    return w


def _seam_mask(y, x, y_positions, x_positions, tile_h, tile_w, out_h, out_w, feather):
    """(1,1,th,tw) centre-cut blend mask for the tile starting at (y, x)."""
    wy = _seam_axis(y_positions, y_positions.index(y), tile_h, out_h, feather)
    wx = _seam_axis(x_positions, x_positions.index(x), tile_w, out_w, feather)
    return (wy[:, None] * wx[None, :])[None, None].clamp_min(1e-6)

def _cosine_mask(h: int, w: int, feather: int, device, dtype) -> torch.Tensor:
    """
    Cosine ramp weight mask (1,1,H,W) for seamless tile blending.
    Edges fade to ~0.0001 so every pixel has a positive denominator.
    """
    if feather <= 0:
        return torch.ones((1, 1, h, w), device=device, dtype=dtype)

    y = torch.ones(h, device=device, dtype=dtype)
    x = torch.ones(w, device=device, dtype=dtype)

    fy = min(feather, h // 2)
    fx = min(feather, w // 2)

    if fy > 0:
        t = torch.linspace(0.0, 1.0, fy, device=device, dtype=dtype)
        ramp = 0.5 - 0.5 * torch.cos(t * math.pi)
        y[:fy] = ramp
        y[-fy:] = ramp.flip(0)

    if fx > 0:
        t = torch.linspace(0.0, 1.0, fx, device=device, dtype=dtype)
        ramp = 0.5 - 0.5 * torch.cos(t * math.pi)
        x[:fx] = ramp
        x[-fx:] = ramp.flip(0)

    return (y[:, None] * x[None, :])[None, None].clamp(1e-4, 1.0)


def _match_color_stats(
    source: torch.Tensor, reference: torch.Tensor, strength: float
) -> torch.Tensor:
    """Reinhard-style per-channel mean/std transfer. Ref is moved to source device."""
    if strength <= 0:
        return source
    dev = source.device
    src = source.float()
    ref = reference.float().to(dev)

    s_mu = src.mean(dim=(2, 3), keepdim=True)
    s_sg = src.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)
    r_mu = ref.mean(dim=(2, 3), keepdim=True)
    r_sg = ref.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)

    matched = (src - s_mu) / s_sg * r_sg + r_mu
    return torch.lerp(src, matched, strength).clamp(0.0, 1.0)


def _contrast_lock(
    source: torch.Tensor, reference: torch.Tensor, strength: float
) -> torch.Tensor:
    """Scale luminance contrast of source toward reference. Ref moved to source device."""
    if strength <= 0:
        return source
    dev = source.device
    ref = reference.float().to(dev)

    src_luma = source.mean(1, keepdim=True)
    ref_luma = ref.mean(1, keepdim=True)

    sc = (src_luma - src_luma.mean((2, 3), keepdim=True)).abs().mean((2, 3), keepdim=True).clamp_min(1e-5)
    rc = (ref_luma - ref_luma.mean((2, 3), keepdim=True)).abs().mean((2, 3), keepdim=True).clamp_min(1e-5)

    factor = (rc / sc).clamp(0.5, 2.0)
    s_mu = source.mean((2, 3), keepdim=True)
    adjusted = (source - s_mu) * factor + s_mu
    return torch.lerp(source, adjusted, strength).clamp(0.0, 1.0)


def _clone_conditioning(cond):
    return copy.deepcopy(cond)


def _apply_controlnet(pos, neg, cn, ctrl_img, strength, start_pct, end_pct):
    if cn is None or strength <= 0 or comfy_nodes is None:
        return pos, neg
    if hasattr(comfy_nodes, "ControlNetApplyAdvanced"):
        return comfy_nodes.ControlNetApplyAdvanced().apply_controlnet(
            pos, neg, cn, ctrl_img, strength, start_pct, end_pct
        )
    if hasattr(comfy_nodes, "ControlNetApply"):
        pos = comfy_nodes.ControlNetApply().apply_controlnet(pos, cn, ctrl_img, strength)[0]
    return pos, neg


def _sample_tile(
    model, pos, neg, vae, tile_img, seed, steps, cfg, sampler, sched, denoise,
):
    """Encode -> KSample -> Decode a single tile image (BHWC)."""
    if comfy_nodes is None:
        return tile_img
    latent_dict = comfy_nodes.VAEEncode().encode(vae, tile_img)[0]
    sampled = comfy_nodes.KSampler().sample(
        model, seed, steps, cfg, sampler, sched, pos, neg, latent_dict, denoise
    )[0]
    return comfy_nodes.VAEDecode().decode(vae, sampled)[0]



def _frequency_structure_lock(model_output, structure_ref, strength, blur_radius=15):
    """Post-process frequency separation structure lock.

    Works with ANY denoise level (including denoise=1.0).
    model_output  = model tile (great HF texture/detail)
    structure_ref = tile_blur or bicubic base (correct LF structure/layout)
    Result: reference structure + model texture -- no ControlNet needed.
    """
    if strength <= 0:
        return model_output
    dev = model_output.device
    ref = structure_ref.float().to(dev)
    out = model_output.float()
    k = max(3, (blur_radius * 2) | 1)
    pad = k // 2
    out_lf = F.avg_pool2d(out, k, stride=1, padding=pad)
    ref_lf = F.avg_pool2d(ref, k, stride=1, padding=pad)
    out_hf = out - out_lf
    blended_lf = torch.lerp(out_lf, ref_lf, strength)
    return (blended_lf + out_hf).clamp(0.0, 1.0)

def _unsharp_mask(image: torch.Tensor, strength: float = 0.5, radius: int = 3) -> torch.Tensor:
    """Unsharp mask sharpening in NCHW format.
    
    - Input/Output: NCHW tensor, float32
    - Tách HF = image - blur(image)
    - Result = image + HF * strength
    - Clamp [0, 1]
    - Kernel size = radius * 2 + 1, đảm bảo odd
    - Dùng F.avg_pool2d cho blur (không cần gaussian)
    - Return image unchanged nếu strength <= 0
    """
    if strength <= 0:
        return image
    k = max(3, (radius * 2) | 1)
    pad = k // 2
    blurred = F.avg_pool2d(image, k, stride=1, padding=pad)
    hf = image - blurred
    return (image + hf * strength).clamp(0.0, 1.0)

def _detail_preserving_color_lock(
    source: torch.Tensor, reference: torch.Tensor,
    color_strength: float, contrast_strength: float,
    blur_radius: int = 7,
) -> torch.Tensor:
    """Color/contrast correction that preserves high-frequency detail.
    
    ALGORITHM:
    1. src_lf = avg_pool(source, kernel)      # low-frequency
    2. src_hf = source - src_lf               # high-frequency (edges, texture)
    3. ref_lf = avg_pool(reference, kernel)    # reference low-frequency
    4. corrected_lf = _match_color_stats(src_lf, ref_lf, color_strength)
    5. corrected_lf = _contrast_lock(corrected_lf, ref_lf, contrast_strength)
    6. return (corrected_lf + src_hf).clamp(0, 1)
    
    KEY INSIGHT: Color correction CHỈ áp dụng lên LF component.
    HF detail (edges, texture, micro-contrast) được giữ nguyên 100%.
    """
    if color_strength <= 0 and contrast_strength <= 0:
        return source
    k = max(3, (blur_radius * 2) | 1)
    pad = k // 2
    src_lf = F.avg_pool2d(source, k, stride=1, padding=pad)
    src_hf = source - src_lf
    ref_lf = F.avg_pool2d(reference.float().to(source.device), k, stride=1, padding=pad)
    
    corrected_lf = src_lf
    if color_strength > 0:
        corrected_lf = _match_color_stats(corrected_lf, ref_lf, color_strength)
    if contrast_strength > 0:
        corrected_lf = _contrast_lock(corrected_lf, ref_lf, contrast_strength)
        
    return (corrected_lf + src_hf).clamp(0.0, 1.0)


def _vegetation_mask(img, threshold=0.05, blur=24):
    """Soft vegetation mask (B,1,H,W) from an NCHW image: normalized excess-green x saturation, box-blurred."""
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    ssum = (r + g + b).clamp_min(1e-4)
    exg = (2.0 * g - r - b) / ssum
    mx = torch.max(torch.max(r, g), b)
    mn = torch.min(torch.min(r, g), b)
    sat = (mx - mn) / mx.clamp_min(1e-4)
    m = ((exg - threshold) / 0.10).clamp(0, 1) * ((sat - 0.08) / 0.10).clamp(0, 1)
    if blur > 0:
        k = (int(blur) * 2) | 1
        m = F.avg_pool2d(m, k, stride=1, padding=k // 2, count_include_pad=False)
        m = F.avg_pool2d(m, k, stride=1, padding=k // 2, count_include_pad=False)
    return m.clamp(0, 1)

def _upscale_with_model(upscale_model, image_bhwc):
    """Use ComfyUI ImageUpscaleWithModel. Returns BHWC or None on failure."""
    # ComfyUI >= 0.3x: V3 node in comfy_extras (classmethod execute -> NodeOutput)
    if upscale_model is not None:
        try:
            from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel as _U
            fn = getattr(_U, "execute", None)
            out = fn(upscale_model, image_bhwc) if fn is not None else _U().upscale(upscale_model, image_bhwc)
            res = getattr(out, "result", out)
            print(f"[DB9] Base upscale with model OK -> {tuple(res[0].shape)}")
            return res[0]
        except Exception as e:
            print(f"[DB9] comfy_extras upscale failed ({e}), trying legacy API.")
    if comfy_nodes is None or upscale_model is None:
        return None
    try:
        # Try both attribute names (ComfyUI version differences)
        upscaler_node = getattr(comfy_nodes, 'ImageUpscaleWithModel', None)
        if upscaler_node is None:
            import nodes as _nodes_module
            upscaler_node = getattr(_nodes_module, 'ImageUpscaleWithModel', None)
        if upscaler_node is None:
            raise AttributeError("ImageUpscaleWithModel not found")
        return upscaler_node().upscale(upscale_model, image_bhwc)[0]
    except Exception as e:
        print(f"[DB9] Upscale model error ({e}), falling back to bicubic.")
        return None


# ---------------------------------------------------------------------------
# Sobel helper (shared)
# ---------------------------------------------------------------------------

def _sobel(gray: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                       dtype=gray.dtype, device=gray.device).unsqueeze(0)
    ky = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
                       dtype=gray.dtype, device=gray.device).unsqueeze(0)
    return torch.sqrt(F.conv2d(gray, kx, padding=1) ** 2 +
                      F.conv2d(gray, ky, padding=1) ** 2 + 1e-6)


# ===========================================================================
# Node: DB9GlobalControlGuide
# ===========================================================================

class DB9GlobalControlGuide:
    """
    Generate a ControlNet guidance image from a source image.
    Supports: sobel, soft_edge, canny, tile_blur, line_art, depth_approx.
    """

    GUIDE_TYPES = ["sobel", "soft_edge", "canny", "tile_blur", "line_art", "depth_approx"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "guide_type": (cls.GUIDE_TYPES,),
                "detect_resolution": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 8.0, "step": 0.1}),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("control_image",)
    FUNCTION = "build"
    CATEGORY = "DB9/Flux Locked Upscale"

    def build(self, image, guide_type, detect_resolution, blur, invert):
        x = _bhwc_to_nchw(image)
        b, c, h, w = x.shape

        scale = detect_resolution / max(h, w)
        nh = max(64, int(round(h * scale)))
        nw = max(64, int(round(w * scale)))
        resized = _resize_nchw(x, (nh, nw), "bilinear")
        gray = resized.mean(1, keepdim=True)

        # tile_blur returns color image -- handle separately
        if guide_type == "tile_blur":
            k = max(3, (max(nh, nw) // 64) * 2 + 1)
            out = F.avg_pool2d(resized, k, stride=1, padding=k // 2)
            out = _resize_nchw(out, (h, w), "bicubic")
            if invert:
                out = 1.0 - out
            return (_nchw_to_bhwc(out.clamp(0, 1)),)

        # Compute edge/depth map
        if guide_type == "sobel":
            edge = _sobel(gray)
        elif guide_type == "soft_edge":
            smooth = F.avg_pool2d(gray, 7, stride=1, padding=3)
            edge = (gray - smooth).abs()
        elif guide_type == "canny":
            s = _sobel(gray)
            s = s / s.amax((2, 3), keepdim=True).clamp_min(1e-6)
            strong = (s > 0.3).float()
            weak = ((s >= 0.1) & (s <= 0.3)).float() * 0.5
            edge = (strong + weak).clamp(0, 1)
        elif guide_type == "line_art":
            smooth = F.avg_pool2d(gray, 3, stride=1, padding=1)
            s = _sobel(smooth)
            edge = 1.0 - (s / s.amax((2, 3), keepdim=True).clamp_min(1e-6)).clamp(0, 1)
        elif guide_type == "depth_approx":
            lf = F.avg_pool2d(gray, 31, stride=1, padding=15)
            lf = lf / lf.amax((2, 3), keepdim=True).clamp_min(1e-6)
            r, g, b_ = resized[:, 0:1], resized[:, 1:2], resized[:, 2:3]
            sat = (torch.max(torch.max(r, g), b_) - torch.min(torch.min(r, g), b_)) / \
                  (torch.max(torch.max(r, g), b_) + 1e-6)
            edge = (lf * 0.7 + sat * 0.3).clamp(0, 1)
        else:
            edge = _sobel(gray)

        # Normalize (skip for types that already produce [0,1])
        if guide_type not in ("canny", "line_art", "depth_approx"):
            edge = edge / edge.amax((2, 3), keepdim=True).clamp_min(1e-6)

        if blur > 0:
            k = max(1, int(round(blur * 2)))
            if k % 2 == 0:
                k += 1
            edge = F.avg_pool2d(edge, k, stride=1, padding=k // 2)

        if invert:
            edge = 1.0 - edge

        edge = _resize_nchw(edge, (h, w), "bicubic")
        if edge.shape[1] == 1:
            edge = edge.repeat(1, 3, 1, 1)
        return (_nchw_to_bhwc(edge.clamp(0, 1)),)


# ===========================================================================
# Node: DB9MiDaSDepth
# ===========================================================================

class DB9MiDaSDepth:
    """
    Real monocular depth estimation using Intel MiDaS (via torch.hub).
    First call downloads the model weights (~100 MB for MiDaS_small).
    Model is cached in memory for subsequent calls.
    """

    MODEL_TYPES = ["MiDaS_small", "DPT_Hybrid", "DPT_Large"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_type": (cls.MODEL_TYPES,),
                "detect_resolution": ("INT", {"default": 384, "min": 128, "max": 1024, "step": 32}),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_map",)
    FUNCTION = "estimate"
    CATEGORY = "DB9/Flux Locked Upscale"

    def estimate(self, image, model_type, detect_resolution, invert):
        global _MIDAS_CACHE

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_str = str(device)

        # Load / reload if model type changed
        if (_MIDAS_CACHE["model"] is None
                or _MIDAS_CACHE["model_type"] != model_type
                or _MIDAS_CACHE["device"] != device_str):
            print(f"[DB9] Loading MiDaS {model_type} on {device_str} ...")
            try:
                midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
                midas.to(device).eval()
                transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
                transform = (transforms.small_transform
                             if model_type == "MiDaS_small"
                             else transforms.dpt_transform)
                _MIDAS_CACHE.update(
                    model=midas, transform=transform,
                    model_type=model_type, device=device_str
                )
                print(f"[DB9] MiDaS {model_type} loaded.")
            except Exception as e:
                print(f"[DB9] MiDaS load failed: {e}. Falling back to depth_approx.")
                return DB9GlobalControlGuide().build(image, "depth_approx", detect_resolution, 0.0, invert)

        midas = _MIDAS_CACHE["model"]
        transform = _MIDAS_CACHE["transform"]

        x = _bhwc_to_nchw(image)
        b, c, h, w = x.shape

        results = []
        for i in range(b):
            # Convert to uint8 numpy HWC (MiDaS transform expects numpy)
            img_np = (x[i].movedim(0, -1).clamp(0, 1).cpu().numpy() * 255).astype("uint8")

            # Resize to detect_resolution keeping aspect (PIL, no cv2 needed)
            scale = detect_resolution / max(h, w)
            nh2 = max(32, int(round(h * scale)))
            nw2 = max(32, int(round(w * scale)))

            try:
                from PIL import Image as _PIL
                import numpy as _np
                pil = _PIL.fromarray(img_np)
                pil = pil.resize((nw2, nh2), _PIL.LANCZOS)
                img_np = _np.array(pil)
            except Exception:
                # torch-only fallback
                tmp = _resize_nchw(x[i:i+1], (nh2, nw2), "bilinear")
                img_np = (tmp[0].movedim(0, -1).cpu().numpy() * 255).astype("uint8")

            inp = transform(img_np).to(device)
            with torch.no_grad():
                pred = midas(inp)
                pred = F.interpolate(pred.unsqueeze(1), size=(h, w),
                                     mode="bicubic", align_corners=False).squeeze(1)

            depth = pred[0].cpu().float()
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
            results.append(depth)

        depth_batch = torch.stack(results).unsqueeze(1).repeat(1, 3, 1, 1)  # (B,3,H,W)
        if invert:
            depth_batch = 1.0 - depth_batch
        return (_nchw_to_bhwc(depth_batch.clamp(0, 1)),)


# ===========================================================================
# Node: DB9FluxControlNetLoader
# ===========================================================================

FLUX_CONTROLNET_PRESETS = {
    "Tile / Upscaler  [jasperai -- recommended for upscale]": {
        "repo_id": "jasperai/Flux.1-dev-Controlnet-Upscaler",
        "filename": "diffusion_pytorch_model.safetensors",
    },
    "Union ControlNet [InstantX -- Canny / Depth / Pose / Tile]": {
        "repo_id": "InstantX/FLUX.1-dev-Controlnet-Union",
        "filename": "diffusion_pytorch_model.safetensors",
    },
    "Canny [XLabs v3]": {
        "repo_id": "XLabs-AI/flux-controlnet-canny-v3",
        "filename": "flux-dev-controlnet-canny.safetensors",
    },
    "Depth [XLabs v3]": {
        "repo_id": "XLabs-AI/flux-controlnet-depth-v3",
        "filename": "flux-dev-controlnet-depth.safetensors",
    },
    "Line Art / HED [XLabs v3]": {
        "repo_id": "XLabs-AI/flux-controlnet-hed-v3",
        "filename": "flux-dev-controlnet-hed.safetensors",
    },
}


class DB9FluxControlNetLoader:
    """
    Download a Flux-compatible ControlNet from HuggingFace and load it
    as a ComfyUI CONTROL_NET object.

    The file is saved to ComfyUI's models/controlnet/ directory so it is
    available in the standard ControlNetLoader too.

    hf_token: required only for gated repos. Leave blank for public models.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(FLUX_CONTROLNET_PRESETS.keys()),),
                "hf_token": ("STRING", {"default": "", "multiline": False}),
                "force_redownload": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("CONTROL_NET",)
    RETURN_NAMES = ("control_net",)
    FUNCTION = "load"
    CATEGORY = "DB9/Flux Locked Upscale"

    def load(self, preset, hf_token, force_redownload):
        if not _HF_AVAILABLE:
            raise RuntimeError(
                "[DB9] huggingface_hub not installed.\n"
                "Run:  pip install huggingface_hub"
            )

        info = FLUX_CONTROLNET_PRESETS[preset]
        repo_id = info["repo_id"]
        filename = info["filename"]
        models_dir = _controlnet_dir()

        # Local destination filename (flat, no subdirs)
        safe_name = repo_id.replace("/", "__") + "__" + filename
        local_path = os.path.join(models_dir, safe_name)

        if force_redownload and os.path.exists(local_path):
            os.remove(local_path)
            print(f"[DB9] Removed cached file: {local_path}")

        if not os.path.exists(local_path):
            print(f"[DB9] Downloading {repo_id}/{filename} ...")
            token = hf_token.strip() or None
            try:
                tmp = _hf_download(
                    repo_id=repo_id,
                    filename=filename,
                    token=token,
                )
                import shutil
                shutil.copy2(tmp, local_path)
                print(f"[DB9] Saved to {local_path}")
            except Exception as e:
                raise RuntimeError(
                    f"[DB9] Download failed for {repo_id}/{filename}\n"
                    f"Error: {e}\n"
                    f"If the model is gated, provide your HuggingFace token in hf_token."
                )
        else:
            print(f"[DB9] Using cached: {local_path}")

        if _comfy_cn is None:
            raise RuntimeError("[DB9] comfy.controlnet not available -- is ComfyUI running?")

        print(f"[DB9] Loading ControlNet ...")
        try:
            cn = _comfy_cn.load_controlnet(local_path)
            return (cn,)
        except Exception as e:
            raise RuntimeError(f"[DB9] Failed to load ControlNet: {e}")


# ===========================================================================
# Node: DB9ColorToneLock
# ===========================================================================

class DB9ColorToneLock:
    """Post-process color/tone correction against a reference image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "color_strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "contrast_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "lock"
    CATEGORY = "DB9/Flux Locked Upscale"

    def lock(self, image, reference, color_strength, contrast_strength):
        src = _bhwc_to_nchw(image)
        ref = _bhwc_to_nchw(reference)
        if src.shape[2:] != ref.shape[2:]:
            ref = _resize_nchw(ref, src.shape[2:], "bicubic")
        out = _match_color_stats(src, ref, color_strength)
        out = _contrast_lock(out, ref, contrast_strength)
        return (_nchw_to_bhwc(out),)


# ===========================================================================
# Node: DB9TileSplit
# ===========================================================================

class DB9TileSplit:
    """
    Split a large image into overlapping tiles for independent processing.

    Output:
      tiles       -- IMAGE batch (N, tile_h, tile_w, C)
      tile_layout -- TILE_LAYOUT dict (pass to DB9TileMerge unchanged)

    Tiles at the edges are reflected-padded to the full tile size so every
    tile going into KSampler has the same shape.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_width": ("INT", {"default": 1536, "min": 256, "max": 4096, "step": 64}),
                "tile_height": ("INT", {"default": 1536, "min": 256, "max": 4096, "step": 64}),
                "overlap": ("INT", {"default": 192, "min": 0, "max": 512, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE", "TILE_LAYOUT")
    RETURN_NAMES = ("tiles", "tile_layout")
    FUNCTION = "split"
    CATEGORY = "DB9/Flux Locked Upscale"

    def split(self, image, tile_width, tile_height, overlap):
        # Always operate on the first image in the batch
        img = image[:1]
        src = _bhwc_to_nchw(img)  # (1, C, H, W)
        _, c, h, w = src.shape

        y_positions = _make_positions(h, tile_height, overlap)
        x_positions = _make_positions(w, tile_width, overlap)

        tiles = []
        tile_rects = []  # (y, x, y2, x2) in output-canvas coords

        for y in y_positions:
            for x in x_positions:
                y2 = min(h, y + tile_height)
                x2 = min(w, x + tile_width)
                th, tw = y2 - y, x2 - x

                tile = src[:, :, y:y2, x:x2]  # (1, C, th, tw)

                # Reflect-pad to uniform tile size
                if th < tile_height or tw < tile_width:
                    tile = F.pad(tile, (0, tile_width - tw, 0, tile_height - th), mode="reflect")

                tiles.append(_nchw_to_bhwc(tile))   # (1, tile_h, tile_w, C)
                tile_rects.append((y, x, y2, x2))

        # (N, tile_h, tile_w, C)
        tiles_batch = torch.cat(tiles, dim=0)

        layout = {
            "out_h": h,
            "out_w": w,
            "tile_height": tile_height,
            "tile_width": tile_width,
            "overlap": overlap,
            "tile_rects": tile_rects,
            "num_tiles": len(tile_rects),
        }

        print(
            f"[DB9] TileSplit: {h}x{w}px -> {len(tile_rects)} tiles "
            f"({len(y_positions)} rows x {len(x_positions)} cols) "
            f"tile={tile_height}x{tile_width} overlap={overlap}"
        )
        return (tiles_batch, layout)


# ===========================================================================
# Node: DB9TileMerge
# ===========================================================================

class DB9TileMerge:
    """
    Merge a batch of processed tiles back into a full-resolution image.

    Tiles are blended with a cosine feather mask so seams are invisible.
    If a tile's spatial size was altered by KSampler/VAE, it is automatically
    bicubic-resized back to the expected slot size before blending.

    Optional reference_image + color_lock_strength applies a global
    color correction at the end.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "tile_layout": ("TILE_LAYOUT",),
                "feather": ("INT", {"default": 128, "min": 0, "max": 512, "step": 8}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "color_lock_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "merge"
    CATEGORY = "DB9/Flux Locked Upscale"

    def merge(self, tiles, tile_layout, feather, reference_image=None, color_lock_strength=0.0):
        out_h = tile_layout["out_h"]
        out_w = tile_layout["out_w"]
        tile_rects = tile_layout["tile_rects"]
        num_tiles = tile_layout["num_tiles"]

        # tiles: (N, tile_h, tile_w, C)
        tiles_nchw = _bhwc_to_nchw(tiles)   # (N, C, tile_h, tile_w)
        dev = tiles_nchw.device
        dtype = tiles_nchw.dtype

        acc = torch.zeros((1, 3, out_h, out_w), device=dev, dtype=dtype)
        wts = torch.zeros((1, 1, out_h, out_w), device=dev, dtype=dtype)

        for i, (y, x, y2, x2) in enumerate(tile_rects):
            if i >= tiles_nchw.shape[0]:
                print(f"[DB9] TileMerge: only {tiles_nchw.shape[0]} tiles available, expected {num_tiles}")
                break

            th, tw = y2 - y, x2 - x
            tile = tiles_nchw[i:i+1]  # (1, C, tile_h, tile_w)

            # Auto-resize if KSampler/VAE changed the spatial size
            if tile.shape[2] != th or tile.shape[3] != tw:
                tile = _resize_nchw(tile, (th, tw), "bicubic")

            mask = _cosine_mask(th, tw, feather, dev, dtype)
            acc[:, :, y:y2, x:x2] += tile * mask
            wts[:, :, y:y2, x:x2] += mask

        out = acc / wts.clamp_min(1e-5)

        if reference_image is not None and color_lock_strength > 0:
            ref = _bhwc_to_nchw(reference_image).to(dev)
            if ref.shape[2:] != (out_h, out_w):
                ref = _resize_nchw(ref, (out_h, out_w), "bicubic")
            out = _match_color_stats(out, ref, color_lock_strength)

        return (_nchw_to_bhwc(out),)


# ===========================================================================
# Node: DB9FluxLockedControlNetUpscale  (all-in-one main node)
# ===========================================================================

class DB9FluxLockedControlNetUpscale:
    """
    All-in-one tiled Flux upscale with:
      - Pixel-space base upscale (bicubic / bilinear / lanczos / upscale model)
      - Per-tile KSampler (Flux img2img)
      - Per-tile + global color & contrast locking
      - Cosine feather seam blending
      - Optional ControlNet guidance per tile
      - Progress bar in ComfyUI UI
      - Console log per tile

    Canvas is kept on CPU so large outputs (6000-8000 px) don't OOM VRAM.
    Sampled tiles are moved to CPU immediately after VAE decode.
    """

    @classmethod
    def INPUT_TYPES(cls):
        samplers = ["euler", "euler_cfg_pp", "dpmpp_2m", "dpmpp_2m_sde", "uni_pc"]
        schedulers = ["normal", "simple", "sgm_uniform", "karras", "beta"]
        if comfy_nodes is not None:
            try:
                samplers = comfy_nodes.KSampler.SAMPLERS
                schedulers = comfy_nodes.KSampler.SCHEDULERS
            except Exception:
                pass

        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05}),
                "tile_width": ("INT", {"default": 1536, "min": 512, "max": 2048, "step": 64}),
                "tile_height": ("INT", {"default": 1536, "min": 512, "max": 2048, "step": 64}),
                "overlap": ("INT", {"default": 192, "min": 0, "max": 768, "step": 8}),
                "feather": ("INT", {"default": 128, "min": 0, "max": 512, "step": 8}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (samplers,),
                "scheduler": (schedulers,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "base_upscale_method": (["bicubic", "bilinear", "lanczos"],),
                "color_lock_strength": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01}),
                "contrast_lock_strength": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sharpen_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.5, "step": 0.05}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),
                "control_net": ("CONTROL_NET",),
                "control_image": ("IMAGE",),
                "control_strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 2.0, "step": 0.01}),
                "control_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "control_end_percent": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                # --- ControlNet-free structure guide (works with any model) ---
                "structure_guide_image": ("IMAGE",),
                "structure_guide_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                # --- v4: vegetation-aware dual denoise + architecture detail lock ---
                "veg_denoise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "veg_mask_threshold": ("FLOAT", {"default": 0.05, "min": -0.2, "max": 0.5, "step": 0.01}),
                "veg_mask_blur": ("INT", {"default": 24, "min": 0, "max": 128, "step": 1}),
                "arch_detail_lock": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "positive_veg": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "base_reference")
    FUNCTION = "upscale"
    CATEGORY = "DB9/Flux Locked Upscale"

    def upscale(
        self,
        image, model, vae, positive, negative,
        scale, tile_width, tile_height, overlap, feather,
        seed, steps, cfg, sampler_name, scheduler, denoise,
        base_upscale_method, color_lock_strength, contrast_lock_strength, sharpen_strength=0.4,
        upscale_model=None,
        control_net=None, control_image=None,
        control_strength=0.65, control_start_percent=0.0, control_end_percent=0.75,
        structure_guide_image=None, structure_guide_strength=0.5,
        veg_denoise=0.0, veg_mask_threshold=0.05, veg_mask_blur=24,
        arch_detail_lock=0.0, positive_veg=None,
    ):
        src = _bhwc_to_nchw(image)
        b, c, h, w = src.shape

        out_h = max(1, int(round(h * scale)))
        out_w = max(1, int(round(w * scale)))

        # -- Step 1: pixel-space base upscale --
        if upscale_model is not None:
            upscaled_bhwc = _upscale_with_model(upscale_model, image)
            if upscaled_bhwc is not None:
                base = _bhwc_to_nchw(upscaled_bhwc)
                if base.shape[2:] != (out_h, out_w):
                    base = _resize_nchw(base, (out_h, out_w), "bicubic")
            else:
                base = _resize_nchw(src, (out_h, out_w), base_upscale_method)
        else:
            base = _resize_nchw(src, (out_h, out_w), base_upscale_method)

        # Move to CPU -- canvas lives on CPU to avoid VRAM OOM for large outputs
        cpu = torch.device("cpu")
        base_cpu = base.to(cpu)

        control_full = base_cpu
        if control_image is not None:
            control_full = _resize_nchw(
                _bhwc_to_nchw(control_image).to(cpu), (out_h, out_w), "bilinear"
            )

        # Structure guide: resize to full output size (will be sliced per tile)
        # WARNING: structure guide causes ghosting when denoise > 0.55
        # because model generates new content at different positions than the reference LF.
        # Auto-disable when denoise is too high.
        structure_full = None
        _effective_sg_strength = structure_guide_strength
        if structure_guide_image is not None and structure_guide_strength > 0:
            # Ghost risk increases with denoise (model generates more different content).
            # Scale structure_guide DOWN as denoise goes up to prevent ghost.
            # Formula: at denoise=0   -> 100% strength (model changes nothing, no ghost risk)
            #          at denoise=0.5 -> 50% strength
            #          at denoise=1.0 -> 0%  strength (model fully rewrites, ghost guaranteed)
            _effective_sg_strength = structure_guide_strength * max(0.0, 1.0 - denoise)
            if _effective_sg_strength < 0.02:
                _effective_sg_strength = 0.0
            if _effective_sg_strength != structure_guide_strength:
                print(
                    f"[DB9] Structure guide: denoise={denoise:.2f} auto-scales "
                    f"strength {structure_guide_strength:.2f} -> {_effective_sg_strength:.2f} "
                    f"(higher denoise = higher ghost risk = lower strength)"
                )
            if _effective_sg_strength > 0:
                structure_full = _resize_nchw(
                    _bhwc_to_nchw(structure_guide_image).to(cpu), (out_h, out_w), "bilinear"
                )
                print(f"[DB9] Structure guide active (effective_strength={_effective_sg_strength:.2f})")

        # -- Step 2: tile positions --
        y_positions = _make_positions(out_h, tile_height, overlap)
        x_positions = _make_positions(out_w, tile_width, overlap)
        total_tiles = len(y_positions) * len(x_positions)

        print(
            f"[DB9] Upscale: {h}x{w} -> {out_h}x{out_w} | "
            f"{total_tiles} tiles ({len(y_positions)}r x {len(x_positions)}c) | "
            f"steps={steps} denoise={denoise}"
        )

        # Progress bar (shows in ComfyUI UI)
        pbar = None
        if _comfy_utils is not None:
            try:
                pbar = _comfy_utils.ProgressBar(total_tiles)
            except Exception:
                pass

        # -- Step 3: tile accumulation canvas (CPU) --
        acc = torch.zeros((b, 3, out_h, out_w), device=cpu, dtype=torch.float32)
        wts = torch.zeros((b, 1, out_h, out_w), device=cpu, dtype=torch.float32)

        tile_index = 0
        _cn_disabled = False  # becomes True if ControlNet fails (shape mismatch)

        for y in y_positions:
            for x in x_positions:
                y2 = min(out_h, y + tile_height)
                x2 = min(out_w, x + tile_width)
                th, tw = y2 - y, x2 - x

                tile_base = base_cpu[:, :, y:y2, x:x2]        # (B,C,th,tw) CPU
                tile_ctrl = control_full[:, :, y:y2, x:x2]    # (B,C,th,tw) CPU
                tile_ref = tile_base.clone()

                tile_image_bhwc = _nchw_to_bhwc(tile_base)    # (B,th,tw,C) CPU
                tile_ctrl_bhwc = _nchw_to_bhwc(tile_ctrl)

                # Keep a clean copy of conditioning for fallback
                tile_pos_clean = _clone_conditioning(positive)
                tile_neg_clean = _clone_conditioning(negative)

                # Apply ControlNet unless already disabled due to earlier error
                tile_pos = _clone_conditioning(positive)
                tile_neg = _clone_conditioning(negative)
                active_cn = None if _cn_disabled else control_net
                tile_pos, tile_neg = _apply_controlnet(
                    tile_pos, tile_neg, active_cn, tile_ctrl_bhwc,
                    control_strength, control_start_percent, control_end_percent,
                )

                # Slice structure guide to this tile (if active)
                tile_structure_bhwc = None
                if structure_full is not None:
                    tile_structure_nchw = structure_full[:, :, y:y2, x:x2]
                    tile_structure_bhwc = _nchw_to_bhwc(tile_structure_nchw)

                try:
                    sampled_bhwc = _sample_tile(
                        model, tile_pos, tile_neg, vae,
                        tile_image_bhwc,
                        int(seed) + tile_index,
                        steps, cfg, sampler_name, scheduler, denoise,
                    )
                except RuntimeError as e:
                    err_str = str(e)
                    if ("cannot be multiplied" in err_str or "mat1" in err_str
                            or "shape" in err_str.lower()) and not _cn_disabled:
                        print(
                            f"[DB9] WARNING: ControlNet shape mismatch — "
                            f"the connected ControlNet is NOT compatible with this model.\n"
                            f"[DB9]   Error: {err_str}\n"
                            f"[DB9]   Disabling ControlNet for all remaining tiles and retrying."
                        )
                        _cn_disabled = True
                        # Retry this tile without ControlNet (but keep structure guide)
                        sampled_bhwc = _sample_tile(
                            model, tile_pos_clean, tile_neg_clean, vae,
                            tile_image_bhwc,
                            int(seed) + tile_index,
                            steps, cfg, sampler_name, scheduler, denoise,
                        )
                    else:
                        raise

                # Critical: force sampled result to CPU canvas device
                sampled_nchw = _bhwc_to_nchw(sampled_bhwc).to(cpu)

                # Auto-resize if VAE changed the spatial dimensions
                if sampled_nchw.shape[2:] != (th, tw):
                    sampled_nchw = _resize_nchw(sampled_nchw, (th, tw), "bicubic")

                # --- v4: architecture detail lock (keep base structure >~7px, take only fine grain from model) ---
                if arch_detail_lock > 0:
                    sampled_nchw = _frequency_structure_lock(sampled_nchw, tile_ref, arch_detail_lock, blur_radius=3)
                # --- v4: vegetation pass (higher denoise, own prompt), blended by soft mask ---
                if veg_denoise > 0:
                    _veg_m = _vegetation_mask(tile_ref, veg_mask_threshold, veg_mask_blur)
                    _veg_cov = float(_veg_m.mean())
                    if _veg_cov >= 0.01:
                        _vp = _clone_conditioning(positive_veg if positive_veg is not None else positive)
                        _vb = _sample_tile(
                            model, _vp, _clone_conditioning(negative), vae, tile_image_bhwc,
                            int(seed) + tile_index, steps, cfg, sampler_name, scheduler, veg_denoise,
                        )
                        _vn = _bhwc_to_nchw(_vb).to(cpu)
                        if _vn.shape[2:] != (th, tw):
                            _vn = _resize_nchw(_vn, (th, tw), "bicubic")
                        sampled_nchw = sampled_nchw * (1.0 - _veg_m) + _vn * _veg_m
                        print(f"[DB9] Tile {tile_index + 1}: vegetation {_veg_cov * 100:.0f}% -> veg pass denoise={veg_denoise}")

                # Per-tile post-processing (detail-preserving: only corrects LF, keeps HF intact)
                if color_lock_strength > 0 or contrast_lock_strength > 0:
                    sampled_nchw = _detail_preserving_color_lock(
                        sampled_nchw, tile_ref,
                        color_lock_strength, contrast_lock_strength,
                        blur_radius=7,
                    )

                # Frequency structure lock
                if structure_full is not None and _effective_sg_strength > 0:
                    tile_struct_nchw = structure_full[:, :, y:y2, x:x2]
                    sampled_nchw = _frequency_structure_lock(
                        sampled_nchw, tile_struct_nchw, _effective_sg_strength,
                        blur_radius=5,  # small blur = only micro-structure, avoids person ghost
                    )

                # Blend into canvas
                mask = _seam_mask(y, x, y_positions, x_positions, tile_height, tile_width, out_h, out_w, feather)
                acc[:, :, y:y2, x:x2] += sampled_nchw * mask
                wts[:, :, y:y2, x:x2] += mask

                tile_index += 1
                if pbar is not None:
                    pbar.update(1)
                cn_tag = " [no CN]" if _cn_disabled else ""
                print(f"[DB9] Tile {tile_index}/{total_tiles}  y={y}:{y2}  x={x}:{x2}{cn_tag}  done")

        # -- Step 4: normalize + final sharpening (NO double color/contrast lock) --
        out = acc / wts.clamp_min(1e-5)

        # Unsharp mask to recover detail lost during cosine-blend averaging
        if sharpen_strength > 0:
            out = _unsharp_mask(out, strength=sharpen_strength, radius=3)
            print(f"[DB9] Applied unsharp mask (strength={sharpen_strength:.2f})")

        print("[DB9] Upscale complete.")
        return (_nchw_to_bhwc(out), _nchw_to_bhwc(base_cpu))


# ===========================================================================
# ComfyUI registration
# ===========================================================================


# ===========================================================================
# Node: DB9ExactColorRestore (v4.1)
# ===========================================================================

def _srgb_to_lab(x):
    """NCHW sRGB [0,1] -> CIE Lab (D65)."""
    x = x.clamp(0.0, 1.0)
    lin = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    r, g, b = lin[:, 0:1], lin[:, 1:2], lin[:, 2:3]
    X = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883

    def f(t):
        return torch.where(t > 0.008856, t.clamp_min(1e-8) ** (1.0 / 3.0), 7.787 * t + 16.0 / 116.0)

    fx, fy, fz = f(X), f(Y), f(Z)
    return torch.cat([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], 1)


def _lab_to_srgb(lab):
    """CIE Lab (D65) -> NCHW sRGB [0,1]."""
    L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    def fi(t):
        return torch.where(t > 0.206893, t ** 3, (t - 16.0 / 116.0) / 7.787)

    X = fi(fx) * 0.95047
    Y = fi(fy)
    Z = fi(fz) * 1.08883
    r = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    bb = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z
    lin = torch.cat([r, g, bb], 1).clamp(0.0, 1.0)
    out = torch.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin.clamp_min(1e-8) ** (1.0 / 2.4) - 0.055)
    return out.clamp(0.0, 1.0)


def _soft_blur(x, radius):
    """Approximate gaussian: two box blurs (edge-normalised)."""
    if radius <= 0:
        return x
    k = (int(radius) * 2) | 1
    x = F.avg_pool2d(x, k, stride=1, padding=k // 2, count_include_pad=False)
    return F.avg_pool2d(x, k, stride=1, padding=k // 2, count_include_pad=False)


def _exact_color_restore(src, ref, luma_radius=16, chroma_radius=4, luma_strength=1.0, chroma_strength=1.0):
    """src/ref: NCHW sRGB, same size. Replace LF luminance + chroma of src by ref, keep HF of src."""
    ls = _srgb_to_lab(src)
    lr = _srgb_to_lab(ref)
    L, Lr = ls[:, 0:1], lr[:, 0:1]
    ab, abr = ls[:, 1:3], lr[:, 1:3]
    if luma_strength > 0:
        L_new = L - _soft_blur(L, luma_radius) + _soft_blur(Lr, luma_radius)
        L = torch.lerp(L, L_new, luma_strength)
    if chroma_strength > 0:
        ab_new = ab - _soft_blur(ab, chroma_radius) + _soft_blur(abr, chroma_radius)
        ab = torch.lerp(ab, ab_new, chroma_strength)
    return _lab_to_srgb(torch.cat([L, ab], 1))


class DB9ExactColorRestore:
    """Lock colours/tones to the original image (Lab). Keeps the fine detail of `image`.

    luma_radius   : px (at output size) above which brightness follows the reference.
    chroma_radius : px above which hue/saturation follow the reference (small = exact colour).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "luma_radius": ("INT", {"default": 16, "min": 0, "max": 256, "step": 1}),
                "chroma_radius": ("INT", {"default": 4, "min": 0, "max": 128, "step": 1}),
                "luma_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chroma_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "restore"
    CATEGORY = "DB9/Flux Locked Upscale"

    def restore(self, image, reference, luma_radius, chroma_radius, luma_strength, chroma_strength):
        src = _bhwc_to_nchw(image).cpu()
        ref = _bhwc_to_nchw(reference).cpu()
        if ref.shape[0] != src.shape[0]:
            ref = ref[:1].expand(src.shape[0], -1, -1, -1)
        if ref.shape[2:] != src.shape[2:]:
            ref = _resize_nchw(ref, src.shape[2:], "bicubic").clamp(0.0, 1.0)
        outs = []
        for i in range(src.shape[0]):
            outs.append(_exact_color_restore(
                src[i:i + 1], ref[i:i + 1], luma_radius, chroma_radius, luma_strength, chroma_strength))
        print(f"[DB9] Exact color restore: luma_r={luma_radius} chroma_r={chroma_radius} "
              f"strength=({luma_strength:.2f},{chroma_strength:.2f})")
        return (_nchw_to_bhwc(torch.cat(outs, 0)),)


NODE_CLASS_MAPPINGS = {
    "DB9ExactColorRestore": DB9ExactColorRestore,
    "DB9GlobalControlGuide": DB9GlobalControlGuide,
    "DB9MiDaSDepth": DB9MiDaSDepth,
    "DB9FluxControlNetLoader": DB9FluxControlNetLoader,
    "DB9ColorToneLock": DB9ColorToneLock,
    "DB9TileSplit": DB9TileSplit,
    "DB9TileMerge": DB9TileMerge,
    "DB9FluxLockedControlNetUpscale": DB9FluxLockedControlNetUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DB9ExactColorRestore": "DB9 Exact Color Restore",
    "DB9GlobalControlGuide": "DB9 Global Control Guide",
    "DB9MiDaSDepth": "DB9 MiDaS Depth",
    "DB9FluxControlNetLoader": "DB9 Flux ControlNet Loader",
    "DB9ColorToneLock": "DB9 Color Tone Lock",
    "DB9TileSplit": "DB9 Tile Split",
    "DB9TileMerge": "DB9 Tile Merge",
    "DB9FluxLockedControlNetUpscale": "DB9 Flux Locked ControlNet Upscale",
}
