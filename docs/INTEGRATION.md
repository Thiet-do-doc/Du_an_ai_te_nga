# Ghép code và làm việc nhóm

## Quy trình Git đề xuất
Dùng repository mới cho dự án AI. Bộ khung chưa được tạo repo hoặc đẩy lên GitHub.
Nhóm trưởng đưa bộ khung lên main, rồi mỗi người tạo một nhánh:
feature/data-khuyen, feature/video-manh, feature/pose-thiet,
feature/model-vo, feature/ui-thien. Mở PR sau khi chạy kiểm tra cục bộ.

Không sửa nhầm repo web cũ. Không commit data video, ảnh người, database,
trọng số lớn, môi trường .venv hoặc thông tin tài khoản.
Lưu dữ liệu ở nơi nhóm được phép truy cập, hướng dẫn cách tải bằng tài liệu.

## Checklist trước khi ghép
1. Code đúng module phụ trách, giữ interface hoặc báo trước thay đổi.
2. Có clip mẫu/đầu vào hợp lệ và đầu ra đủ trường.
3. Đã chạy `python main.py --demo` và `python -m unittest discover -s tests -v`.
4. Với mô-đun thật, kiểm tra thêm bằng dữ liệu thật; test demo không thay thế test AI.
5. Không nạp lại YOLO/model mỗi frame; không dùng Mock trong demo nghiệm thu.
6. Cập nhật README nếu thêm dependency, lệnh chạy, trọng số hoặc bước tải dữ liệu.

## Thay mô-đun giả bằng thật
1. Mạnh triển khai iter_frames.
2. Thiết triển khai PoseExtractor.extract; kiểm tra đầu ra trước khi chạy cả dataset.
3. Thiết + Võ triển khai vectorizer và chính sách nhãn cửa sổ; khóa hợp đồng.
4. Võ triển khai train/evaluate và TrainedPredictor; model load một lần.
5. Nhóm trưởng tạo entrypoint chạy thật, kết nối frame → pose → buffer → predictor → alerts/store.
   Dùng dict[(source_id,track_id)] cho buffer; reset buffer và AlertManager khi mất track.
6. Thiện dùng chung pipeline đó trong app.py, bổ sung snapshot/âm thanh/lịch sử.

Khung hiện không có camera loop thật hoặc UI hoàn chỉnh. Không đổi tên MockPredictor
thành mô hình thật: phải thay logic suy luận và kiểm thử bằng model đã huấn luyện.

## Báo cáo thực nghiệm
Ghi cấu hình máy, phiên bản thư viện, trọng số pose, seed và split.
So sánh mô hình cùng test/same labels, báo cả lỗi và giới hạn.
Định nghĩa cách ghép dự đoán thành event, cách tính phát hiện đúng và độ trễ.
Video test liên tục có đủ hoạt động dễ nhầm: ngồi nhanh, cúi nhặt đồ, nằm xuống,
ra khỏi khung hình và che khớp. Chỉ mô phỏng chuyển động an toàn, không yêu cầu ngã nguy hiểm.
