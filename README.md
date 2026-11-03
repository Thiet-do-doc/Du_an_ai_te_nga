# An Tâm — Giám sát an toàn

Ứng dụng Streamlit có bốn màn hình:
- **Giám sát trực tiếp:** webcam trên máy chạy ứng dụng (nguồn `0`) hoặc camera IP qua RTSP/HTTP; bắt đầu, dừng, phân tích các đoạn 32 khung hình và lưu lịch sử.
- **Xem lại sự cố:** lọc ngày/kết quả, phát lại và tải video, đánh dấu đã kiểm tra.
- **Báo cáo & thống kê:** số đoạn phân tích, cảnh báo, cảnh báo chưa kiểm tra, dữ liệu chưa đủ; biểu đồ theo ngày và xuất CSV.
- **Tải video lên:** phân tích MP4, AVI, MOV và lưu kết quả cùng video.

## Tải và chạy trên máy khác

Cài **Python 3.11 (64 bit)** và Git. Trên Windows, mở PowerShell rồi chạy:

```powershell
git clone https://github.com/nguyendinhkhuyen/DuAnPhatHienCanhBaoTeNga.git
cd DuAnPhatHienCanhBaoTeNga
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Mở địa chỉ `http://localhost:8501` nếu trình duyệt chưa tự mở. Các lần sau chỉ cần chạy lệnh cuối trong thư mục dự án. Nhấn `Ctrl+C` trong PowerShell để tắt ứng dụng.

Nếu không dùng Git, chọn **Code → Download ZIP** trên GitHub, giải nén, mở PowerShell tại thư mục chứa `app.py`, rồi chạy các lệnh từ bước tạo `.venv` trở đi.

Hai tệp mô hình trong `models/` đã có sẵn trong repo. Máy không có GPU NVIDIA vẫn có thể chạy bằng CPU, nhưng xử lý có thể chậm hơn. Lần cài đầu cần Internet để tải các thư viện.

Video và lịch sử phân tích trong `data/` không được đưa lên Git. Nếu muốn chuyển cả lịch sử cũ, sao chép riêng thư mục này sang máy mới.

Video và cơ sở dữ liệu SQLite được lưu trong `data/` trên máy chạy ứng dụng. Các đoạn camera được mã hóa H.264 để phát lại trong trình duyệt. Video tải lên cần có codec được trình duyệt hỗ trợ để xem trực tiếp.

Camera giám sát chỉ chạy khi màn hình Giám sát trực tiếp đang mở; chuyển sang mục khác sẽ dừng camera. Nhấn Dừng trước khi đóng trang. Đây không phải dịch vụ giám sát chạy nền. Mỗi đoạn được phân tích gồm 32 khung hình lấy theo tốc độ xử lý; video phát lại ở 5 fps nên không thể dùng thời lượng phát lại để đo thời gian thực. Mô hình hiện phân tích người có khung bao lớn nhất; không theo dõi từng người riêng biệt. Số cảnh báo là số đoạn video, không phải số sự cố độc lập. AI hỗ trợ phát hiện dấu hiệu té ngã và cần người kiểm tra lại.
