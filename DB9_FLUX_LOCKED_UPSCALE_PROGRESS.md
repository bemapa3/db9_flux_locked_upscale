# DB9 Flux Locked Upscale — Progress & Handoff
**Date:** 2026-05-20
**Repo:** https://github.com/bemapa3/db9_flux_locked_upscale
**Last commit:** `99fa75d`

---

## Mục tiêu dự án

Tạo một bộ custom nodes ComfyUI chuyên dùng để **upscale hình ảnh bằng Flux 2 Klein 9B** (model nhỏ ~9B, dùng cho Colab).

Các vấn đề cần giải quyết:
1. Tile bị crop sai — chỉ góc trái trên chạy đúng, còn lại chồng lên nhau
2. Màu sắc / ánh sáng / contrast bị drift sau khi upscale
3. Node chạy mãi không ra kết quả (không có progress)
4. Cần port để cắm upscale model (Real-ESRGAN, v.v.)
5. Cần tách/ghép tile thủ công (cho hình 6000px+)
6. Cần depth map thật (không phải approximate)
7. ControlNet crash với Flux 2 Klein 9B
8. Cần cách giữ cấu trúc hình mà không cần ControlNet

---

## Kiến trúc hiện tại — 7 nodes

```
DB9 Global Control Guide         — tạo guide image từ hình gốc (edge, depth, blur...)
DB9 MiDaS Depth                  — depth map thật bằng Intel MiDaS
DB9 Flux ControlNet Loader       — tải ControlNet từ HuggingFace
DB9 Color Tone Lock              — chỉnh màu/contrast theo reference
DB9 Tile Split                   — cắt hình thành batch tiles
DB9 Tile Merge                   — ghép tiles lại thành hình đầy đủ
DB9 Flux Locked ControlNet Upscale — node chính: upscale toàn bộ tự động
```

---

## Chi tiết từng node

---

### 1. DB9 Global Control Guide
**Mục đích:** Tạo ảnh control/guide từ hình gốc, không cần preprocessor ngoài.

**Input:**
- `image` — hình nguồn
- `guide_type` — loại guide: `sobel`, `soft_edge`, `canny`, `tile_blur`, `line_art`, `depth_approx`
- `detect_resolution` — độ phân giải xử lý (default 1024)
- `blur` — blur thêm sau khi tạo guide
- `invert` — đảo ngược

**Guide types giải thích:**
| Type | Dùng khi nào |
|------|-------------|
| `tile_blur` | Upscale — giữ màu sắc/structure tổng thể (khuyến nghị) |
| `sobel` | Edge detection cơ bản |
| `canny` | Edge rõ nét hơn, phù hợp line art |
| `soft_edge` | Edge mượt, không hard |
| `line_art` | Giữ đường nét trắng/đen |
| `depth_approx` | Depth nhanh không cần model |

**Output:** `control_image` — IMAGE có thể cắm vào `structure_guide_image` hoặc `control_image`

---

### 2. DB9 MiDaS Depth
**Mục đích:** Tạo depth map thật bằng Intel MiDaS — dùng được cho ControlNet depth.

**Cách hoạt động:**
- Download model qua `torch.hub` lần đầu chạy (~100MB)
- Cache trong RAM, không download lại
- Fallback về `depth_approx` nếu load lỗi

**Input:**
- `model_type` — `MiDaS_small` (nhanh), `DPT_Hybrid`, `DPT_Large` (chậm hơn, chính xác hơn)
- `detect_resolution` — độ phân giải xử lý (default 384)
- `invert` — đảo depth (gần=tối / xa=sáng, tùy ControlNet)

**Output:** `depth_map` — IMAGE

---

### 3. DB9 Flux ControlNet Loader
**Mục đích:** Tải ControlNet cho Flux từ HuggingFace và load vào ComfyUI.

**QUAN TRỌNG — Giới hạn tương thích:**
- Tất cả ControlNet trong preset đều được train cho **flux.1-dev** (hidden dim = 3072)
- **Flux 2 Klein 9B** có hidden dim = 512 → **KHÔNG tương thích**
- Node vẫn tải về và load được, nhưng khi run sẽ bị shape mismatch error
- Node chính đã có auto-fallback: nếu ControlNet crash, tự retry không có ControlNet

**Presets:**
| Preset | Dùng cho |
|--------|---------|
| Tile/Upscaler (jasperai) | Upscale — flux.1-dev |
| Union ControlNet (InstantX) | Canny/Depth/Pose/Tile tổng hợp — flux.1-dev |
| Canny/Depth/Line Art (XLabs v3) | Chuyên biệt — flux.1-dev |

**Input:** `preset`, `hf_token` (để trống nếu model public), `force_redownload`
**Output:** `control_net`

---

### 4. DB9 Color Tone Lock
**Mục đích:** Post-process chỉnh màu/contrast của output theo reference image.

**Cách hoạt động (Reinhard color transfer):**
```
matched = (output - output_mean) / output_std * ref_std + ref_mean
result = lerp(output, matched, color_strength)
```
Không cần model — thuần toán học, nhanh, không mất detail.

**Input:** `image`, `reference`, `color_strength` (0-1), `contrast_strength` (0-1)
**Output:** `image`

---

### 5. DB9 Tile Split
**Mục đích:** Cắt hình lớn thành batch tiles để xử lý riêng từng tile.

**Cách hoạt động:**
- Tính vị trí tile theo `_make_positions()` — đảm bảo mọi pixel đều được cover
- Tile cuối hàng/cột được reflect-pad để đúng size
- Trả về `tile_layout` dict chứa metadata (toạ độ, size) để TileMerge biết cách ghép lại

**Input:** `image`, `tile_width` (default 1536), `tile_height` (default 1536), `overlap` (default 192)
**Output:** `tiles` (IMAGE batch N×H×W×C), `tile_layout` (TILE_LAYOUT)

**Overlap giải thích:**
- Overlap = vùng chồng giữa 2 tiles kề nhau
- Cosine feather khi merge sẽ blend vùng overlap → xóa seam
- Overlap quá nhỏ → seam rõ. Overlap quá lớn → xử lý chậm
- Khuyến nghị: overlap = feather * 1.5

---

### 6. DB9 Tile Merge
**Mục đích:** Ghép batch tiles lại thành hình hoàn chỉnh với cosine feather blending.

**Cách hoạt động:**
```
Với mỗi tile i tại vị trí (y, x, y2, x2):
  tile → auto-resize về (y2-y, x2-x) nếu VAE đã thay đổi kích thước
  mask = cosine_ramp(feather) — giá trị 0 ở biên, 1 ở giữa
  acc[y:y2, x:x2] += tile * mask
  weights[y:y2, x:x2] += mask

output = acc / weights  — weighted average, seam biến mất
```

**Input:** `tiles`, `tile_layout`, `feather` (default 128), `reference_image` (optional), `color_lock_strength` (optional)
**Output:** `image`

---

### 7. DB9 Flux Locked ControlNet Upscale (Node chính)
**Mục đích:** All-in-one — upscale tự động với tiling, color lock, structure guide.

**Pipeline bên trong:**
```
1. Base upscale (bicubic / lanczos / upscale model) → hình to
2. Chia thành tiles
3. Với mỗi tile:
   a. Apply ControlNet conditioning (nếu có, có auto-fallback)
   b. KSampler → VAE decode
   c. Force tensor về CPU (fix device mismatch bug)
   d. Auto-resize nếu VAE thay đổi kích thước
   e. Color lock vs tile gốc
   f. Contrast lock vs tile gốc
   g. Frequency structure lock vs structure guide (nếu có)
   h. Blend vào canvas với cosine mask
4. Global color/contrast correction vs base
5. Output hình hoàn chỉnh + base_reference
```

**Inputs chính:**
| Input | Mô tả | Default |
|-------|--------|---------|
| `scale` | Hệ số phóng to | 2.0 |
| `tile_width/height` | Kích thước tile | 1536 |
| `overlap` | Vùng chồng | 192 |
| `feather` | Bán kính blend | 128 |
| `denoise` | Mức denoise KSampler | 1.0 |
| `cfg` | CFG scale | 1.0 |
| `color_lock_strength` | Giữ màu vs gốc | 0.55 |
| `contrast_lock_strength` | Giữ contrast vs gốc | 0.20 |
| `upscale_model` (opt) | Real-ESRGAN, v.v. | None |
| `control_net` (opt) | ControlNet (flux.1-dev only) | None |
| `structure_guide_image` (opt) | Guide image cho frequency lock | None |
| `structure_guide_strength` (opt) | Độ mạnh của structure lock | 0.5 |

---

## Tính năng đặc biệt: Frequency Structure Lock

**Vấn đề:** ControlNet không tương thích với Flux 2 Klein 9B.
**Giải pháp:** Dùng frequency separation post-process.

**Nguyên lý:**
Mỗi ảnh có thể tách thành 2 lớp:
- **Low Frequency (LF):** Structure tổng thể — layout, khối màu, ánh sáng chính
- **High Frequency (HF):** Detail bề mặt — texture, sharpness, grain

```
output = lerp(output_LF, reference_LF, strength) + output_HF
       = reference structure          + model texture
```

**Ưu điểm:**
- Không cần model phụ, không cần ControlNet
- Tương thích với MỌI model (Flux, SD, v.v.)
- Hoạt động với mọi `denoise` kể cả `denoise=1.0`
- Giữ được texture/sharpness của model
- Enforce structure từ reference image

**Cách dùng:**
```
DB9GlobalControlGuide (tile_blur) → structure_guide_image
                                          ↓
                         DB9FluxLockedControlNetUpscale
                         └─ structure_guide_strength = 0.4~0.7
```

---

## Bug đã fix

| Bug | Root cause | Fix |
|-----|-----------|-----|
| Tile bị distort, chỉ góc trái đúng | Tensor trên CUDA, canvas trên CPU → mismatch | `.to(cpu)` sau mỗi tile sample |
| Node chạy mãi không output | Không có progress indicator | `comfy.utils.ProgressBar` + print từng tile |
| ControlNet crash + segfault | Flux 2 Klein dim=512 vs ControlNet dim=3072 | Auto-fallback: retry tile không có ControlNet, disable CN từ tile tiếp |

---

## Workflow được khuyến nghị

### Workflow A — Simple (output ≤ 3000px)
```
LoadImage → DB9FluxLockedControlNetUpscale → SaveImage
```
Settings: scale=2.0, tile=1536, overlap=192, denoise=1.0, cfg=1.0

### Workflow B — Structure guide (không cần ControlNet)
```
LoadImage → DB9GlobalControlGuide(tile_blur) ──→ structure_guide_image
LoadImage ──────────────────────────────────→ DB9FluxLockedControlNetUpscale → SaveImage
```
Settings: structure_guide_strength=0.5

### Workflow C — Tile split/merge (output 6000px+)
```
LoadImage → upscale pixel → DB9TileSplit → KSampler → DB9TileMerge → SaveImage
                                 └─ tile_layout ──────────────────────┘
```

---

## Settings khuyến nghị — Flux 2 Klein 9B

| Parameter | Value | Lý do |
|-----------|-------|--------|
| `denoise` | 1.0 | Flux cần full denoise để chất lượng tốt |
| `cfg` | 1.0 | Flux không dùng CFG kiểu SDXL |
| `tile_width/height` | 1536 | Optimal cho Flux 2 Klein 9B (< 2048px) |
| `overlap` | 192 | Đủ rộng để feather blend không thấy seam |
| `feather` | 128 | Cosine ramp 128px — invisible seam |
| `color_lock` | 0.45~0.65 | Giữ màu gốc, không drift |
| `contrast_lock` | 0.15~0.25 | Giữ tương phản |
| `structure_guide` | 0.4~0.6 | Giữ cấu trúc không cần ControlNet |

---

## Next steps gợi ý

- [ ] Test frequency structure lock trên ảnh thật — so sánh với/không có structure guide
- [ ] Test DB9MiDaSDepth trên Colab (torch.hub download hoạt động không)
- [ ] Tìm ControlNet được train cho Flux 2 Klein 9B (hiện chưa có)
- [ ] Thêm `DB9TileControlPrep` — tự động tạo control image per-tile trong split
- [ ] Cân nhắc giảm `color_lock_strength * 0.75` global pass — có thể quá mạnh
