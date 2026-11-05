# DB9 Flux Locked Upscale — STATE / handoff

Cập nhật: 2026-11-06 02:20 (Asia/Saigon)

## Version hiện tại
- `v4.4.0` (tag v4.4.0) — gồm v4.2.0, v4.3.0.
- **CHƯA PUSH GitHub**: phiên Claude cloud không có quyền ghi repo (git proxy 403; tài khoản GitHub chưa liên kết với Claude).
  Repo đã cập nhật đầy đủ (commit + tag) trên Drive: `G:\My Drive\SD-Data\_DB9_Release\db9_flux_locked_upscale`.
  Push từ máy chủ: copy thư mục đó sang `J:\!!_BAOTAPCODE\db9_flux_locked_upscale` rồi
  `git push origin main --follow-tags` (nhánh `backup-c21fa44` chỉ là bản STATE cũ, không cần push).

## Ảnh test / workflow
- Input: `SD-Data/ComfyUIinput/image (8).png` (2000×1500).
- Output `SD-Data/Export/ComfyUI/2026-11-05/`: `BBB-Upscale_v440__00001_.png` = v4.4.0 chốt;
  `BBB-Upscale_v43__00001_.png` = v4.3; `SoSanh_goc_v43_v440.jpg` = so sánh crop.
- Workflow: `workflows/DB9_FluxLockedUpscale_v4.4.0_2026-11-06_api.json` (API format).
  veg_denoise 0.48, veg_blur_boost 0, tone_match 1.0, vram_mode auto. Phần còn lại như v4.3.

## Đã xử lý (v4.4)
- Lá "ghost"/quầng trời ở mép tán: cân tần số thấp lượt cây về lượt chính + nới mask chỉ vào trời.
- Ám xanh lá vùng tối: tone_match theo dải sáng (đo ở ~2k).
- Cây nền nét hơn: veg_denoise 0.48.
- Color Restore chạy GPU (6k: 5 phút → vài giây). Render 6k trên A100: ~14 phút.

## Đã thử và loại
- Nới mask cây ra mọi hướng (mảng vuông trên trời, bẩn góc mái).
- veg_blur_boost (lượt phụ) — trộn 2 lượt vẽ khác nhau tạo lá trong suốt; để mặc định 0.
- KJNodes ColorMatch hm-mvgd-hm ở 6k — lệch xanh dương vùng sáng (ΔE max 3.9).
- Prompt "deep depth of field" — không tác dụng.

## Colab
- Tick **DriveSyn** trước cell 1; bấm cho phép Google Drive.
- Đổi sang **A100** (rẻ hơn G4; dùng ~35–40 GB VRAM).
- Node DB9: `cp -r SD-Data/_DB9_Release/db9_flux_locked_upscale /content/ComfyUI/custom_nodes/` rồi chạy lại cell RUN.
  (Đã có trong `Custom_ComfyNode.txt` dạng git clone từ GitHub — chỉ đúng bản mới sau khi push.)
- Dán lệnh dài vào terminal Colab: gõ phím bị cắt ~2000 ký tự; dùng sự kiện paste qua JS.

## Ý tưởng tiếp theo
- Chế độ "realism" kiểu MIR (chủ hỏi https://www.mir.no/work): cây cỏ denoise 0.55–0.65, công trình 0.30–0.35,
  khoá màu nhẹ hơn. Cần chủ gửi ảnh và mức "được phép vẽ lại".
- Test thật trên RTX 5070 Ti (fp8) và 5090.

## Bắt buộc mỗi lần đổi code
- Đo + so crop 100 % đúng vùng chủ khoanh trước khi báo xong.
- Commit tác giả Claude <noreply@anthropic.com>, ghi CHANGELOG + versions.json.
