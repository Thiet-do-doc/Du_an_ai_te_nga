# AI phát hiện té ngã qua camera — Nhóm 7

Bộ khung Python dành cho 5 thành viên phát triển độc lập và ghép code theo giao diện chung.
Đây là skeleton, chưa chứa mô hình đã huấn luyện hoặc chức năng nhận diện camera thật.
Demo sử dụng điểm dự đoán giả, chỉ xác minh luồng gom chuỗi → dự đoán → xác nhận → SQLite.

## Chạy ngay (Python 3.11, không cần cài thư viện)

Giải nén, mở thư mục `fall_detection_ai` bằng VS Code rồi mở terminal tại thư mục đó:

```bash
python main.py --demo
python -m unittest discover -s tests -v
```

Demo tạo `outputs/demo_alerts.sqlite3`; mỗi lần chạy tái hiện một sự kiện giả.
Không dùng kết quả demo để đánh giá độ chính xác phát hiện té ngã.

## Bản đồ dự án

| Đường dẫn | Vai trò | Phụ trách |
|---|---|---|
| `src/fall_ai/dataset.py` | Kiểm kê dữ liệu, nhãn, chia tập theo người/sự kiện | Khuyến |
| `src/fall_ai/video.py` | Đọc video/camera, lấy mẫu theo timestamp | Mạnh |
| `src/fall_ai/pose.py` | YOLO-Pose, theo dõi người, chuẩn hóa khớp | Thiết |
| `src/fall_ai/sequence.py` | Bộ đệm chuỗi theo từng người | Thiết + Võ |
| `src/fall_ai/training.py` | Random Forest, LSTM, đánh giá và xuất mô hình | Võ |
| `src/fall_ai/predictor.py` | Giao diện suy luận và bộ giả lập | Võ |
| `src/fall_ai/alerts.py` | Xác nhận sự kiện, thời gian chờ | Thiện + Võ |
| `src/fall_ai/storage.py` | Lưu cảnh báo SQLite | Thiện |
| `app.py` | Điểm vào giao diện Streamlit, chưa triển khai | Thiện |
| `main.py` | Điểm vào demo tích hợp | Thiết / nhóm trưởng |
| `src/fall_ai/contracts.py` | Kiểu dữ liệu dùng chung | Cả nhóm thống nhất |
| `config/default.json` | Cấu hình thử nghiệm chung | Nhóm trưởng |
| `docs/` | Hợp đồng dữ liệu, kế hoạch, hướng dẫn tích hợp | Cả nhóm |
| `data/raw/` | Video gốc, không đưa lên Git | Khuyến |
| `data/processed/` | Đặc trưng đã xử lý, không đưa lên Git | Thiết |
| `models/` | Trọng số và metadata, không đưa lên Git | Võ |
| `outputs/` | Kết quả thực nghiệm, DB và ảnh, không đưa lên Git | Cả nhóm |

## Đọc trước khi code

1. `docs/ARCHITECTURE.md`: luồng offline và online.
2. `docs/CONTRACTS.md`: định dạng dữ liệu và nhãn.
3. `docs/TEAM_PLAN.md`: việc cần làm, bàn giao, tiêu chí hoàn thành.
4. `docs/INTEGRATION.md`: quy trình nhánh Git và tích hợp.

## Thư viện cho giai đoạn triển khai thật

Tạo môi trường riêng trước khi cài: `python -m venv .venv`.
Windows PowerShell: `.venv\Scripts\Activate.ps1`.
Windows CMD: `.venv\Scripts\activate.bat`.
Linux/macOS: `source .venv/bin/activate`.

`requirements-optional.txt` là danh sách dự kiến, chưa khóa phiên bản và chưa được
kiểm thử tương thích trong bộ khung này. Chỉ cài phần cần dùng; cả nhóm chốt môi trường
và lưu phiên bản sau khi có bản tích hợp hoạt động. PyTorch chọn bản phù hợp CPU/GPU.
Khóa phiên bản YOLO-Pose và trọng số giống nhau khi trích xuất và chạy thật.

Không gọi hotline hoặc gửi thông báo ra ngoài. Phạm vi bản đầu: một người, camera cố định,
video có sẵn/webcam, cảnh báo cục bộ. Xin phép người tham gia trước khi ghi/lưu hình ảnh.
