import copy
import math

import torch
import torch.nn.functional as F

try:
    import nodes
except Exception:
    nodes = None


def _bhwc_to_nchw(image):
    if image.ndim != 4:
        raise ValueError("ComfyUI IMAGE must be BHWC")
    return image.movedim(-1, 1).float()


def _nchw_to_bhwc(image):
    return image.movedim(1, -1).clamp(0.0, 1.0)


def _resize_nchw(x, size, mode="bicubic"):
    if mode == "nearest":
        return F.interpolate(x, size=size, mode=mode)
    if mode == "lanczos":
        mode = "bicubic"
    return F.interpolate(x, size=size, mode=mode, align_corners=False)


def _make_positions(size, tile, overlap):
    if size <= tile:
        return [0]
    step = max(1, tile - overlap)
    positions = list(range(0, size - tile + 1, step))
    last = size - tile
    if positions[-1] != last:
        positions.append(last)
    return positions


def _cosine_mask(h, w, feather, device, dtype):
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
        y[-fy:] = torch.flip(ramp, dims=[0])

    if fx > 0:
        t = torch.linspace(0.0, 1.0, fx, device=device, dtype=dtype)
        ramp = 0.5 - 0.5 * torch.cos(t * math.pi)
        x[:fx] = ramp
        x[-fx:] = torch.flip(ramp, dims=[0])

    return (y[:, None] * x[None, :])[None, None, :, :].clamp(0.001, 1.0)


def _match_color_stats(source, reference, strength):
    if strength <= 0:
        return source

    src = source.float()
    ref = reference.float()

    src_mean = src.mean(dim=(2, 3), keepdim=True)
    src_std = src.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)

    ref_mean = ref.mean(dim=(2, 3), keepdim=True)
    ref_std = ref.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)

    matched = (src - src_mean) / src_std * ref_std + ref_mean
    return torch.lerp(src, matched, strength).clamp(0.0, 1.0)


def _contrast_lock(source, reference, strength):
    if strength <= 0:
        return source

    src_luma = source.mean(dim=1, keepdim=True)
    ref_luma = reference.mean(dim=1, keepdim=True)

    src_center = src_luma - src_luma.mean(dim=(2, 3), keepdim=True)
    ref_center = ref_luma - ref_luma.mean(dim=(2, 3), keepdim=True)

    src_contrast = src_center.abs().mean(dim=(2, 3), keepdim=True).clamp_min(1e-5)
    ref_contrast = ref_center.abs().mean(dim=(2, 3), keepdim=True).clamp_min(1e-5)

    factor = (ref_contrast / src_contrast).clamp(0.5, 2.0)
    adjusted = (source - source.mean(dim=(2, 3), keepdim=True)) * factor + source.mean(dim=(2, 3), keepdim=True)
    return torch.lerp(source, adjusted, strength).clamp(0.0, 1.0)


def _clone_conditioning(cond):
    return copy.deepcopy(cond)


def _apply_controlnet(positive, negative, control_net, control_image, strength, start_percent, end_percent):
    if control_net is None or strength <= 0:
        return positive, negative

    if nodes is None:
        return positive, negative

    if hasattr(nodes, "ControlNetApplyAdvanced"):
        applier = nodes.ControlNetApplyAdvanced()
        return applier.apply_controlnet(
            positive,
            negative,
            control_net,
            control_image,
            strength,
            start_percent,
            end_percent,
        )

    if hasattr(nodes, "ControlNetApply"):
        applier = nodes.ControlNetApply()
        positive = applier.apply_controlnet(positive, control_net, control_image, strength)[0]
        return positive, negative

    return positive, negative


def _sample_tile(model, positive, negative, vae, tile_image, seed, steps, cfg, sampler_name, scheduler, denoise):
    if nodes is None:
        return tile_image

    vae_encoder = nodes.VAEEncode()
    sampler = nodes.KSampler()
    vae_decoder = nodes.VAEDecode()

    latent = vae_encoder.encode(vae, tile_image)[0]

    sampled = sampler.sample(
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent,
        denoise,
    )[0]

    decoded = vae_decoder.decode(vae, sampled)[0]
    return decoded


class DB9GlobalControlGuide:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "guide_type": (["sobel", "soft_edge"],),
                "detect_resolution": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                }),
                "blur": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 8.0,
                    "step": 0.1,
                }),
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
        gray = resized.mean(dim=1, keepdim=True)

        if guide_type == "sobel":
            kx = torch.tensor(
                [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                dtype=gray.dtype,
                device=gray.device,
            ).unsqueeze(0)
            ky = torch.tensor(
                [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
                dtype=gray.dtype,
                device=gray.device,
            ).unsqueeze(0)

            gx = F.conv2d(gray, kx, padding=1)
            gy = F.conv2d(gray, ky, padding=1)
            edge = torch.sqrt(gx * gx + gy * gy + 1e-6)
        else:
            smooth = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3)
            edge = (gray - smooth).abs()

        edge = edge / edge.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)

        if blur > 0:
            k = max(1, int(round(blur * 2)))
            if k % 2 == 0:
                k += 1
            edge = F.avg_pool2d(edge, kernel_size=k, stride=1, padding=k // 2)

        if invert:
            edge = 1.0 - edge

        edge = edge.repeat(1, 3, 1, 1)
        return (_nchw_to_bhwc(edge),)


class DB9ColorToneLock:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "color_strength": ("FLOAT", {
                    "default": 0.65,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "contrast_strength": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
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


class DB9FluxLockedControlNetUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        samplers = ["euler", "euler_cfg_pp", "dpmpp_2m", "dpmpp_2m_sde", "uni_pc"]
        schedulers = ["normal", "simple", "sgm_uniform", "karras", "beta"]

        if nodes is not None:
            try:
                samplers = nodes.KSampler.SAMPLERS
                schedulers = nodes.KSampler.SCHEDULERS
            except Exception:
                pass

        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "scale": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 8.0,
                    "step": 0.05,
                }),
                "tile_width": ("INT", {
                    "default": 1536,
                    "min": 512,
                    "max": 2048,
                    "step": 64,
                }),
                "tile_height": ("INT", {
                    "default": 1536,
                    "min": 512,
                    "max": 2048,
                    "step": 64,
                }),
                "overlap": ("INT", {
                    "default": 192,
                    "min": 0,
                    "max": 768,
                    "step": 8,
                }),
                "feather": ("INT", {
                    "default": 128,
                    "min": 0,
                    "max": 512,
                    "step": 8,
                }),
                "seed": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                }),
                "cfg": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                }),
                "sampler_name": (samplers,),
                "scheduler": (schedulers,),
                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "base_upscale_method": (["bicubic", "bilinear", "lanczos"],),
                "color_lock_strength": ("FLOAT", {
                    "default": 0.55,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "contrast_lock_strength": ("FLOAT", {
                    "default": 0.20,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
            },
            "optional": {
                "control_net": ("CONTROL_NET",),
                "control_image": ("IMAGE",),
                "control_strength": ("FLOAT", {
                    "default": 0.65,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                }),
                "control_start_percent": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "control_end_percent": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "base_reference")
    FUNCTION = "upscale"
    CATEGORY = "DB9/Flux Locked Upscale"

    def upscale(
        self,
        image,
        model,
        vae,
        positive,
        negative,
        scale,
        tile_width,
        tile_height,
        overlap,
        feather,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        base_upscale_method,
        color_lock_strength,
        contrast_lock_strength,
        control_net=None,
        control_image=None,
        control_strength=0.65,
        control_start_percent=0.0,
        control_end_percent=0.75,
    ):
        src = _bhwc_to_nchw(image)
        b, c, h, w = src.shape

        out_h = max(1, int(round(h * scale)))
        out_w = max(1, int(round(w * scale)))

        base = _resize_nchw(src, (out_h, out_w), base_upscale_method)

        if control_image is None:
            control_full = base
        else:
            control_full = _resize_nchw(_bhwc_to_nchw(control_image), (out_h, out_w), "bilinear")

        y_positions = _make_positions(out_h, tile_height, overlap)
        x_positions = _make_positions(out_w, tile_width, overlap)

        acc = torch.zeros_like(base)
        weights = torch.zeros((b, 1, out_h, out_w), device=base.device, dtype=base.dtype)

        tile_index = 0

        for y in y_positions:
            for x in x_positions:
                y2 = min(out_h, y + tile_height)
                x2 = min(out_w, x + tile_width)

                tile_base = base[:, :, y:y2, x:x2]
                tile_ctrl = control_full[:, :, y:y2, x:x2]
                tile_ref = tile_base

                tile_image = _nchw_to_bhwc(tile_base)
                tile_control_image = _nchw_to_bhwc(tile_ctrl)

                tile_pos = _clone_conditioning(positive)
                tile_neg = _clone_conditioning(negative)

                tile_pos, tile_neg = _apply_controlnet(
                    tile_pos,
                    tile_neg,
                    control_net,
                    tile_control_image,
                    control_strength,
                    control_start_percent,
                    control_end_percent,
                )

                tile_seed = int(seed) + tile_index
                sampled = _sample_tile(
                    model,
                    tile_pos,
                    tile_neg,
                    vae,
                    tile_image,
                    tile_seed,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    denoise,
                )

                sampled_nchw = _bhwc_to_nchw(sampled)

                if sampled_nchw.shape[2:] != tile_base.shape[2:]:
                    sampled_nchw = _resize_nchw(sampled_nchw, tile_base.shape[2:], "bicubic")

                sampled_nchw = _match_color_stats(sampled_nchw, tile_ref, color_lock_strength)
                sampled_nchw = _contrast_lock(sampled_nchw, tile_ref, contrast_lock_strength)

                mask = _cosine_mask(
                    sampled_nchw.shape[2],
                    sampled_nchw.shape[3],
                    feather,
                    sampled_nchw.device,
                    sampled_nchw.dtype,
                )

                acc[:, :, y:y2, x:x2] += sampled_nchw * mask
                weights[:, :, y:y2, x:x2] += mask

                tile_index += 1

        out = acc / weights.clamp_min(1e-5)
        out = _match_color_stats(out, base, color_lock_strength * 0.75)
        out = _contrast_lock(out, base, contrast_lock_strength * 0.75)

        return (_nchw_to_bhwc(out), _nchw_to_bhwc(base))


NODE_CLASS_MAPPINGS = {
    "DB9GlobalControlGuide": DB9GlobalControlGuide,
    "DB9ColorToneLock": DB9ColorToneLock,
    "DB9FluxLockedControlNetUpscale": DB9FluxLockedControlNetUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DB9GlobalControlGuide": "DB9 Global Control Guide",
    "DB9ColorToneLock": "DB9 Color Tone Lock",
    "DB9FluxLockedControlNetUpscale": "DB9 Flux Locked ControlNet Upscale",
}