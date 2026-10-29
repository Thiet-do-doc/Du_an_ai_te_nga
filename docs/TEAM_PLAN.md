# Kế hoạch dành cho nhóm trưởng

## Bàn giao theo 5 thành viên
| Thành viên | Bắt đầu | Bàn giao | Tiêu chí nhận |
|---|---|---|---|
| Khuyến | Tải/xem dữ liệu, xác nhận nguồn và nhãn | Manifest, thống kê, split | Không trùng người/sự kiện giữa tập; đếm được nhãn |
| Mạnh | Đọc 1 video | iter_frames, webcam | Timestamp đúng; giải phóng camera; xử lý file lỗi |
| Thiết | Pose trên 1 clip | PoseExtractor, vectorizer, đặc trưng | 17 khớp, mask, bbox; track nhất quán; offline/online cùng chuẩn hóa |
| Võ | Đọc hợp đồng, thử input giả | Baseline, LSTM, Predictor, báo cáo | Chạy cùng split; nạp được model; không lộ nhãn test |
| Thiện | UI với dữ liệu giả | Streamlit, SQLite, cảnh báo | Hiển thị trạng thái thiếu dữ liệu; không lưu trùng theo frame |

## Bốn mốc triển khai (ước lượng, điều chỉnh theo lịch nộp)
1. Kết nối kỹ thuật: 1 video qua toàn bộ pipeline; dùng predictor giả có nhãn rõ.
2. Dữ liệu + baseline: manifest chuẩn, split, trích xuất, huấn luyện Random Forest.
3. LSTM + camera: so sánh validation, nạp mô hình thật, luồng webcam và UI.
4. Nghiệm thu: test giữ riêng, video liên tục, báo cáo, video demo, hướng dẫn chạy.

Không đợi toàn bộ dữ liệu để ghép mô-đun lần đầu. Không đợi LSTM để bắt đầu UI.

## Việc nhóm trưởng theo dõi
- Chốt hợp đồng dữ liệu trước, review các thay đổi vào contracts/config.
- Duy trì main chạy được; mỗi nhánh chỉ ghép sau khi có mẫu đầu ra và hướng dẫn.
- Mỗi người cập nhật: đã làm, đầu ra ở đâu, đang vướng, cần ai hỗ trợ.
- Ghép thử thường xuyên trên máy trình bày, ghi lại cấu hình và phiên bản.
- Báo cáo phân biệt phần đã chạy, đang làm, dự kiến; không dùng kết quả mock làm thực nghiệm.

## Backlog gợi ý
- DATA-01: kiểm kê/EDA và xác nhận ý nghĩa nhãn.
- DATA-02: split theo người/sự kiện và chính sách nhãn cửa sổ.
- VIDEO-01: đọc video timestamp + lấy mẫu.
- VIDEO-02: webcam và xử lý mất kết nối.
- POSE-01: 17 khớp + confidence + track.
- POSE-02: chuẩn hóa/vectorizer dùng chung + lưu chuỗi.
- ML-01: Random Forest + báo cáo validation.
- ML-02: LSTM + checkpoint + metadata.
- ML-03: Predictor thật + so sánh công bằng + test cuối.
- UI-01: khung Streamlit và hiển thị video/trạng thái.
- ALERT-01: xác nhận/cooldown/phục hồi và lưu SQLite.
- INTEGRATION-01: video thật qua toàn luồng.
- QA-01: video liên tục, báo giả/giờ, độ trễ, mất người.
- REPORT-01: tổng hợp báo cáo, demo và hướng dẫn cài.
