# Kiến trúc dự án

## Bài toán
Phân loại chuỗi chuyển động thành té ngã/không té ngã; demo camera một người,
trong nhà, đủ ánh sáng. Phân biệt té ngã với chủ động nằm xuống bằng diễn biến thời gian.
YOLO-Pose là bộ trích đặc trưng pretrained; Random Forest/LSTM là mô hình nhóm tự huấn luyện.

## Offline — Khuyến → Mạnh → Thiết → Võ
1. Manifest: đường dẫn, người, phiên/sự kiện, hoạt động, nhãn, thời điểm té ngã.
2. Tách train/validation/test theo người; các bản quay cùng sự kiện luôn cùng tập.
3. Đọc video và lấy mẫu theo timestamp, không chỉ dùng cùng bước nhảy cho mọi FPS.
4. Trích 17 khớp, mask, confidence và bbox.
5. Tạo cửa sổ 30 mẫu ở 15 Hz (mốc thử ban đầu, điều chỉnh bằng validation).
6. Tạo đặc trưng thống kê cho Random Forest, chuỗi cho LSTM.
7. Huấn luyện; chọn model và ngưỡng bằng validation; báo cáo test sau khi chốt.
8. Xuất model + metadata về schema, nhãn, chuẩn hóa, FPS, cửa sổ, phiên bản YOLO.

## Online — Mạnh → Thiết → Võ → Thiện
Video/webcam → frame có timestamp → pose theo track → bộ đệm riêng cho từng track
→ mô hình → bộ xác nhận → giao diện + SQLite.
Mô hình không đọc trực tiếp từ SQLite. UI không tự huấn luyện lại mô hình.
Đọc/xử lý frame một nơi; UI hiển thị kết quả từ cùng luồng xử lý.

Mất người: xóa bộ đệm và reset trạng thái xác nhận cho track đó.
Mở lại video: tạo lại trạng thái nguồn, vì timestamp video trở về 0.
Không thấy người không đồng nghĩa với không té ngã: UI hiển thị thiếu dữ liệu.

## Trạng thái hiện tại
Đã có: kiểu dữ liệu, bộ đệm, cơ chế xác nhận đơn giản, SQLite, demo giả lập và test.
Chưa có: dữ liệu thật, đọc camera, YOLO-Pose, huấn luyện, suy luận thật, giao diện, ảnh/âm thanh cảnh báo.
Demo không đo accuracy, FPS camera hay độ trễ mô hình thật.

## Điều kiện nghiệm thu cuối
Đọc được video và webcam; giữ đúng người; dùng cùng tiền xử lý offline/online.
Có hai mô hình so sánh trên cùng tập test và cùng chính sách gán nhãn.
Báo cáo Precision/Recall/F1 lớp té ngã, confusion matrix, FPS và cấu hình máy.
Thử video liên tục để đo phát hiện theo sự kiện, báo giả/giờ và độ trễ.
Ghi nhận lỗi ở tình huống cúi/nhặt đồ/ngồi nhanh/nằm xuống, che khuất và mất người.
Không cam kết số liệu chất lượng trước khi thực nghiệm.
