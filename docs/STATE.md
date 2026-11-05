# DB9 Flux Locked Upscale — STATE / handoff

Cập nhật: 2026-11-05 23:30 (Asia/Saigon)

## Version hiện tại
- `v4.3.0` (commit e673684, tag v4.3.0) — có cả v4.2.0 (b2717fa).
- **CHƯA PUSH GitHub** (phiên Claude cloud bị chặn quyền ghi repo, lỗi 403 git proxy).
  Push từ máy chủ: copy thư mục repo này sang `J:\!!_BAOTAPCODE\db9_flux_locked_upscale`
  rồi `git push origin main --follow-tags`.

## Ảnh test
- Input: `SD-Data/ComfyUIinput/image (8).png` (2000×1500, villa + sân golf).
- Output: `SD-Data/Export/ComfyUI/2026-11-05/`
  - `BBB-UpscaleF2K9B__00011_.png` = v4.2 tốt nhất
  - `BBB-Upscale_v43__00001_.png` = v4.3
- Workflow: `workflows/DB9_FluxLockedUpscale_v4.3.0_2026-11-05_api.json` (API format, kéo thả vào ComfyUI).
  Seed 287922623580276, denoise 0.28, veg_denoise 0.42, arch_detail_lock 0.5, texture_limit 1.25,
  veg_include_flowers on, Nomos8kSC, tile 1536/overlap 192/feather 128, LoRA DB9_Upscale_Ver004 0.5.

## Colab
- Notebook SDVN_ComfyUI: **tick DriveSyn** trước khi chạy cell 1 (bản update 04/11 để mặc định tắt).
- Domain Pinggy cố định đã hết hạn; token cũ ở `SD-Data/Setting/Domain_sever.txt.bak`. Link free 60 phút.
- Cloudflare link thường 403 với trình duyệt của chủ → dùng link Pinggy.
- Custom node trên Colab cần clone lại mỗi phiên (hoặc copy từ repo này).

## Vấn đề chủ đang báo (mở)
1. Cây góc trên-trái (tán xanh đậm phía sau, giữa cây lớn và bụi hoa trắng, ~x 300–900, y 800–1400 ở ảnh 6k) vẫn nhoè.
2. Tone màu tổng thể hơi lệch xanh lá → cần khoá màu chính xác hơn.
3. Tường xám (mảng ốp ~x 1400–1600, y 1150–1400) còn vân rạn nhẹ — base Nomos có thể làm nổi.

## Đã kiểm tra 2026-11-05 23:15
- Vùng cây nhoè (ảnh 6k ~x150–750, y750–1250): mask cây = 0.99 → không phải lỗi mask/seam.
  Nguyên nhân: lượt cây (prompt "raw photo") cho Flux làm mờ hậu cảnh kiểu bokeh/DOF.
  Thử: thêm "deep depth of field, entire scene in sharp focus, f/11, no bokeh, no background blur"
  vào prompt nhà (node 9) và prompt cây (node 36) → render `BBB-Upscale_v43_dof_*`
  (prompt API lưu ở `/content/db9_v43b_prompt.json` trên Colab).
- Màu: đo Lab so với ảnh gốc — toàn ảnh, vùng cây, hoa, cỏ, tường đều lệch < 0.2 (không có
  bias xanh). Không có ICC profile ở input/output. Ở thang chi tiết 6 px chroma ΔE ≈ 2.0
  (do vị trí lá khác, không phải lệch tông).
  - chroma_radius 8 → ΔE chi tiết 0.94 nhưng **quầng vàng quanh lá quay lại** → loại.
  - Giảm chroma chi tiết (k 0.6–0.75) → không giảm ΔE → loại.
  → Cần hỏi chủ đang so với ảnh/ứng dụng nào khi thấy "ngả xanh".

- Kết quả prompt "deep DOF / no bokeh" (`BBB-Upscale_v43_dof__00001_.png`, 470 s): **không có tác dụng**
  (HF energy cây nền 0.0485 → 0.0481, nhìn giống hệt). Cây nền đã mờ sẵn trong ảnh gốc (ảnh render
  có DOF); veg_denoise 0.42 giữ nguyên độ mờ đó.

## 2026-11-06 00:10 — v4.4 (wip, commit a260cb8, chưa tag)
- v4.4  `veg_blur_boost` (0.10) / `veg_blur_fraction` (0.35): lượt thứ 3 cho phần cây mờ nhất (denoise 0.42+0.10).
- v4.4b `tone_match` (1.0) trong Exact Color Restore: khớp a/b theo từng dải sáng (chủ thấy gốc ám tím, bản upscale ám xanh lá).
- v4.4c mask cây nới thêm ~24 px để lá mảnh ở mép tán trên nền trời chỉ lấy từ 1 lượt → hết lá "ghost" (chủ khoanh đỏ).
- Colab đổi sang **A100 80GB** (6.88 đv/giờ, rẻ hơn G4 10.05). Chậm: ~1.7 s/bước, 6k mất ~35 phút với vram_mode `high (32GB)`.
- Render test: `BBB-Upscale_v44_*` (script queue `/content/q44.py`, đo `/content/m44.py` — mất khi tắt runtime;
  bản copy ở scratchpad Claude). Cài nodes.py v4.4 lên Colab: copy repo Drive vào custom_nodes rồi ghi đè nodes.py (md5 64cc948d…).
- Lưu ý vram_mode giá trị đầy đủ: 'auto','low (<=16GB)','medium (24GB)','high (32GB)','xhigh (40-48GB)','ultra (80GB+)'.

## Ý tưởng tiếp theo
- Làm nét cây nền: chạy riêng vùng cây nền với veg_denoise cao hơn (0.55–0.6) bằng mask "nền mờ"
  (vùng cây có HF energy thấp trong ảnh base) → cần chống bịa chi tiết; hoặc hỏi chủ có muốn giữ
  độ sâu trường ảnh như ảnh gốc không.
- texture_limit so với ảnh gốc phóng bicubic thay vì base Nomos.
- Exact Color Restore: chroma radius nhỏ hơn (8–12) vẫn edge-aware; có thể thêm khớp màu theo vùng mask cây riêng.
- Kiểm tra mask/seam tại vùng cây nhoè (đường ghép tile y≈1440, x≈1440).

## Bắt buộc mỗi lần đổi code
- Đo + so crop 100% đúng vùng chủ khoanh trước khi báo xong.
- Commit tác giả Claude <noreply@anthropic.com>, ghi CHANGELOG + versions.json.
