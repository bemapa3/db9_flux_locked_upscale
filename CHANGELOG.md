# Changelog

## v4.2.0 — 2026-11-05
### ✅ Đã làm
- **Sửa lỗi `upscale_model` không chạy** trên ComfyUI ≥ 0.3x: `ImageUpscaleWithModel` đã chuyển sang `comfy_extras.nodes_upscale_model` (node V3), code cũ không tìm thấy nên âm thầm phóng base bằng bicubic. Nay gọi đúng API mới (giữ đường cũ làm dự phòng). Kiểm chứng trên ComfyUI 0.44: log `Base upscale with model OK -> (1, 6000, 8000, 3)`.
- **v4 — tách xử lý công trình / cây cỏ** trong `DB9FluxLockedControlNetUpscale` (input tuỳ chọn, mặc định tắt nên workflow cũ không đổi):
  - `arch_detail_lock`: giữ cấu trúc ảnh base từ ~7 px trở lên, chỉ nhận texture mịn từ Flux → vật liệu không bị vẽ thêm hoa văn.
  - `veg_denoise`, `veg_mask_threshold`, `veg_mask_blur`, `positive_veg`: mask cây cỏ tự động (excess-green × độ bão hoà) + lượt sample thứ hai cho lá/cỏ với prompt riêng, trộn theo mask mềm.
- **v4.1 — node mới `DB9 Exact Color Restore`**: khoá màu theo ảnh gốc trong không gian Lab (độ sáng tần số thấp + sắc độ từ ảnh gốc, chi tiết tần số cao giữ từ ảnh upscale). Đo trên ảnh test 2k → 6k: lệch màu cục bộ ΔE 4.61 → 0.31, lệch trung bình (L/a/b) −0.43/+0.39/−0.31 → −0.14/0.00/−0.02.
- **v4.2 — ghép tile bằng đường cắt giữa vùng chồng**: trước đây cả vùng chồng bị trộn trung bình hai tile có texture khác nhau → dải nhoè dọc đường ghép (tile cuối hàng/cột chồng tới 1000+ px). Nay mỗi pixel thuộc một tile, chỉ dải `feather` px ở giữa được cross-fade (trọng số cộng đúng 1). Đo năng lượng chi tiết vùng cây cỏ: vùng 4 tile chồng / vùng 1 tile từ 0.61 → 0.96.
- README: thêm tham số mới, node mới, preset đã test cho kiến trúc + cảnh quan 2k → 6k.

### ⏳ Chưa làm / chưa kiểm chứng
- Mask cây cỏ chỉ nhận cây xanh; cây hoa tím/hồng được xử lý như công trình.
- `DB9 Tile Merge` vẫn dùng kiểu trộn cosine cũ cho cả vùng chồng.
- Chưa test trên ComfyUI bản cũ (< 0.3x) cho đường gọi `upscale_model` dự phòng.
- Chưa test với ảnh nội thất / ảnh không có cây.

## v3.0.0 — 2026-06-01
- Rewrite v3: bỏ double color/contrast lock, thêm color lock giữ chi tiết, unsharp mask sau khi ghép, canvas trên CPU, progress bar; thêm DB9MiDaSDepth, DB9FluxControlNetLoader, DB9TileSplit, DB9TileMerge.
