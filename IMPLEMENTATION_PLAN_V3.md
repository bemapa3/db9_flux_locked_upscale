# DB9 Flux Locked Upscale — V3 Implementation Plan
## Mục tiêu: Fix Detail Blur trên chi tiết nhỏ (aerial photos)

**Kỹ sư chính:** Gemini  
**QA / Review:** Codex (Claude)  
**Ngày:** 2026-06-01  
**File cần sửa:** `nodes.py` (v2 → v3)  
**Backup v2:** `nodes_v2_backup.py` (đã tạo sẵn)

---

## Bối cảnh vấn đề

Khi upscale ảnh aerial (chụp từ trên cao) có nhiều chi tiết nhỏ (nhà cửa, thuyền, đường phố), kết quả output bị **blur/mềm** ở các chi tiết nhỏ, dù thông số đã hợp lý (cfg=1, denoise=0.5, steps=12).

### Root Cause Analysis (3 bugs)

| # | Bug | File/Line | Mức độ |
|---|-----|-----------|--------|
| 1 | **Double Color/Contrast Lock** — áp dụng `_match_color_stats` + `_contrast_lock` **2 lần** (per-tile ở L996-1000 VÀ global ở L1024-1027), tổng tác động ~96%, xóa sạch micro-contrast | `nodes.py:996-1000` + `nodes.py:1024-1027` | CRITICAL |
| 2 | **Color lock phá hủy HF detail** — `_match_color_stats` (Reinhard transfer) thay đổi mean/std trên toàn bộ pixel, bao gồm cả high-frequency edges/texture | `nodes.py:158-174` (cách sử dụng) | HIGH |
| 3 | **Không có sharpening pass** — Sau khi merge tiles bằng cosine mask (trung bình hóa vùng overlap), không có bước nào khôi phục lại sharpness | `nodes.py:1021-1028` | MEDIUM |

---

## Chỉ thị thực hiện cho Gemini

### TASK 1: Thêm helper `_unsharp_mask` (NEW)

**Vị trí:** Sau hàm `_frequency_structure_lock` (sau dòng 247)

**Yêu cầu:**
```python
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
```

**Tại sao:** Cosine mask blending tại vùng overlap sẽ trung bình hóa 2 tile có seed khác nhau → blur chi tiết. Unsharp mask khôi phục edges.

---

### TASK 2: Thêm helper `_detail_preserving_color_lock` (NEW)

**Vị trí:** Ngay sau `_unsharp_mask`

**Yêu cầu:**
```python
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
```

**Tại sao:** Reinhard transfer trên full image sẽ scale cả HF detail, làm mất micro-contrast. Frequency separation giữ nguyên HF.

---

### TASK 3: Thêm parameter `sharpen_strength` vào `DB9FluxLockedControlNetUpscale`

**File:** `nodes.py`

**3a. Trong `INPUT_TYPES` (khoảng dòng 815-817):**
Thêm sau `contrast_lock_strength`:
```python
"sharpen_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.5, "step": 0.05}),
```

**3b. Trong method signature `upscale()` (khoảng dòng 842):**
Thêm `sharpen_strength=0.4` vào default args.

---

### TASK 4: Sửa per-tile color lock (MODIFY)

**Vị trí:** `nodes.py` dòng 996-1000

**Code hiện tại (XÓA):**
```python
# Per-tile post-processing
if color_lock_strength > 0:
    sampled_nchw = _match_color_stats(sampled_nchw, tile_ref, color_lock_strength)
if contrast_lock_strength > 0:
    sampled_nchw = _contrast_lock(sampled_nchw, tile_ref, contrast_lock_strength)
```

**Code mới (THAY THẾ):**
```python
# Per-tile post-processing (detail-preserving: only corrects LF, keeps HF intact)
if color_lock_strength > 0 or contrast_lock_strength > 0:
    sampled_nchw = _detail_preserving_color_lock(
        sampled_nchw, tile_ref,
        color_lock_strength, contrast_lock_strength,
        blur_radius=7,
    )
```

---

### TASK 5: Xóa global double lock, thêm unsharp mask (MODIFY)

**Vị trí:** `nodes.py` dòng 1021-1028

**Code hiện tại (XÓA):**
```python
# -- Step 4: normalize + global color/contrast correction --
out = acc / wts.clamp_min(1e-5)

if color_lock_strength > 0:
    out = _match_color_stats(out, base_cpu, color_lock_strength * 0.75)
if contrast_lock_strength > 0:
    out = _contrast_lock(out, base_cpu, contrast_lock_strength * 0.75)
```

**Code mới (THAY THẾ):**
```python
# -- Step 4: normalize + final sharpening (NO double color/contrast lock) --
out = acc / wts.clamp_min(1e-5)

# Unsharp mask to recover detail lost during cosine-blend averaging
if sharpen_strength > 0:
    out = _unsharp_mask(out, strength=sharpen_strength, radius=3)
    print(f"[DB9] Applied unsharp mask (strength={sharpen_strength:.2f})")
```

**CRITICAL: Hoàn toàn XÓA BỎ global `_match_color_stats` và `_contrast_lock` ở đây. Đây chính là bug chính gây blur.**

---

### TASK 6: Update version header

**Dòng 1-3:** Đổi `Rewrite v2 — 2026-05-20` → `Rewrite v3 — 2026-06-01`

**Dòng 5-9:** Update changelog:
```
v3 Fixes (detail blur on small structures):
   - Removed destructive double color/contrast lock (was applied per-tile AND global)
   - Added detail-preserving color lock: extracts HF detail before correction, re-adds after
   - Added unsharp mask post-merge for sharpness recovery
   - Canvas kept on CPU to avoid VRAM OOM on large outputs
   - Progress bar via comfy.utils.ProgressBar
   - Per-tile print so user can see progress in console
```

---

## Verification Checklist (QA — Codex sẽ review)

- [ ] `_unsharp_mask` có guard `strength <= 0` → return unchanged
- [ ] `_detail_preserving_color_lock` có guard cả 2 strength <= 0 → return unchanged
- [ ] Kernel size luôn odd (k % 2 == 0 → k += 1)
- [ ] Global double lock (L1024-1027 cũ) đã bị XÓA HOÀN TOÀN
- [ ] `sharpen_strength` xuất hiện trong INPUT_TYPES, method signature, và Step 4
- [ ] Tất cả tensor operations đều dùng `.clamp(0.0, 1.0)` ở output
- [ ] Không import thêm module mới (chỉ dùng torch, F đã có sẵn)
- [ ] `nodes_v2_backup.py` không bị thay đổi
- [ ] Syntax check: `python -c "import ast; ast.parse(open('nodes.py').read())"` pass
- [ ] Không thay đổi NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS
- [ ] Tất cả existing nodes vẫn hoạt động (backward compatible)
- [ ] `sharpen_strength` có default value trong method signature (backward compat với workflow cũ)

---

## Tóm tắt thay đổi

| Hành động | Dòng cũ | Mô tả |
|-----------|---------|-------|
| ADD | sau L247 | `_unsharp_mask()` helper |
| ADD | sau helper trên | `_detail_preserving_color_lock()` helper |
| ADD | ~L817 | `sharpen_strength` parameter trong INPUT_TYPES |
| ADD | ~L842 | `sharpen_strength` trong method signature |
| REPLACE | L996-1000 | Per-tile color lock → detail-preserving version |
| DELETE+REPLACE | L1021-1028 | Xóa global double lock, thêm unsharp mask |
| MODIFY | L1-9 | Version header update |
