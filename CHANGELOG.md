# Changelog

## v4.4.0 — 2026-11-06
### ✅ Đã làm
- **Cây nền nhoè** — `veg_blur_boost` / `veg_blur_fraction` (mặc định tắt; preset 0.10 / 0.35): tìm 35 % vùng cây kém nét nhất trong ảnh base, chạy thêm một lượt với denoise `veg_denoise + veg_blur_boost` (0.42 → 0.52) và trộn theo mask mềm. Ảnh test: 15–22 % ảnh được tăng nét; vùng cây nền góc trái (chủ khoanh) nét hơn v4.3 khi so crop 100 %.
- **Hết quầng sáng / "ghost" ở mép tán cây trên nền trời** — trước khi trộn, tần số thấp (~24 px: tông trời, bóng đổ) của lượt cây được cân về đúng lượt chính, chỉ giữ chi tiết lá; mask cây được nới ~28 px **chỉ vào vùng trời** (sáng, ngả xanh dương, phẳng) để lá mảnh ở mép tán lấy trọn từ một lượt. So crop 100 % (vùng chủ khoanh đỏ): hết mảng trời sáng quanh tán, lá mờ bán trong suốt giảm rõ; góc mái/tường không bị ảnh hưởng.
- **Khớp tông màu theo dải sáng** — `tone_match` (mặc định 1.0) trong DB9 Exact Color Restore: so trung bình a/b từng dải độ sáng (tối → sáng) giữa ảnh mới và ảnh gốc, đo ở ~2k để độ nét khác nhau không làm lệch thống kê. Ảnh test: bóng tối sâu nhất lệch xanh lá a −1.0 → −0.5, vùng sáng giữ ≈ 0 (bản thử đầu đo ở 6k làm vùng sáng ngả xanh dương −1.4 b — đã sửa).
- Thử và **bỏ**: nới mask cây 24 px ra mọi hướng (v4.4c) → tạo mảng vuông sáng trên trời và vẽ bẩn lên góc mái be.
- Colab: chạy được trên A100 80 GB với `vram_mode` = `high (32GB)` (giả lập RTX 5090): batch 2, VAE không chia ô, 6000×4500 ≈ 20–27 phút (A100 chậm hơn RTX PRO 6000 ~2.5 lần).

### ⏳ Chưa làm / chưa kiểm chứng
- Vẫn còn vài lá mờ nhẹ ở mép tán (do Flux vẽ lá lệch vị trí so với ảnh gốc).
- Dải sáng trung bình (L 60–70) vẫn lệch nhẹ a +0.7 / b −0.8.
- `veg_blur_boost` tăng ~30–40 % thời gian ở các tile có cây mờ.
- Chưa chạy thật trên RTX 5070 Ti / 5090.

## v4.3.0 — 2026-11-05
### ✅ Đã làm
- **Exact Color Restore v2** (sửa viền vàng/xanh quanh lá trên nền trời và lá bị "bóng ma"): chỉ chỉnh màu/độ sáng ở tần số thấp (mặc định `luma_radius` 32, `chroma_radius` 24), trường hiệu chỉnh được lọc bằng guided filter theo ảnh mới (`edge_aware`, mặc định bật) nên bám theo mép lá mới. Kiểm chứng bằng mắt trên vùng cây/trời: hết quầng vàng.
- **Mask cây cỏ v2**: nhận cả lá trong bóng râm (ngưỡng bão hoà theo độ sáng), lấp lỗ và nới mép, chuyển tiếp hẹp. Vùng trộn nửa vời (mask 0.1–0.9) giảm từ 11.8% → 2.4% diện tích ảnh test → hết hiện tượng "chồng hai lớp" ở tán cây.
- `veg_include_flowers` (mặc định tắt): nhận thêm hoa trắng/hồng (sáng, ít bão hoà, có texture, không phải cạnh thẳng kiến trúc; loại cát bằng tỉ lệ xanh dương/xanh lá). Ảnh test: bụi hoa trái 0.97, mặt tiền 0.00, trời 0.00, lòng cát 0.09.
- `texture_limit` (mặc định 0 = tắt, khuyên 1.25): vùng công trình không được có năng lượng chi tiết lớn hơn N lần ảnh base tại chỗ → giảm hoa văn bịa (vân rạn trên tường, vết nứt trên cát).
- **Tối ưu VRAM**: `vram_mode` (auto/low/medium/high/xhigh/ultra) + `tile_batch`. Auto đọc VRAM GPU, giới hạn kích thước tile, đo bộ nhớ thực tế của tile đầu rồi chọn batch; GPU ≤ 24 GB dùng VAE chia ô; gặp OOM tự giảm batch rồi bật VAE chia ô. Đường xử lý mới dùng khi không cắm ControlNet (có ControlNet thì chạy đường cũ).
- Kiểm chứng trên Colab RTX PRO 6000 (95 GB): mode=ultra, ~1.34 GB/tile, batch=6, 6000×4500 trong 503 s. Mode `low` (giả lập 16 GB): tile 1024, VAE chia ô, batch 1, chạy hết không lỗi.

### ⏳ Chưa làm / chưa kiểm chứng
- Chưa chạy thật trên RTX 5070 Ti (16 GB) và 5090 (32 GB). Lưu ý: Flux2-klein-9b bản bf16 nặng ~17 GB, trên 16 GB nên dùng bản fp8.
- Gộp batch không làm nhanh hơn trên RTX PRO 6000 (GPU đã chạy hết công suất với 1 tile 1536); lợi ích chính của `vram_mode` là tránh tràn VRAM trên card nhỏ.
- Tường xám vẫn còn vân rạn nhẹ (đã giảm so với v4.2); có thể do chính ảnh base Nomos8kSC làm nổi vân. Hướng tiếp theo: so `texture_limit` với ảnh gốc phóng bicubic thay vì ảnh base.
- Lượt cây cỏ vẫn có thể vẽ thêm vài chi tiết nhỏ (đốm hoa trong cỏ).


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
