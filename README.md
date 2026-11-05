# FallGuard AI — Phát hiện té ngã

Ứng dụng Streamlit dùng YOLO11n-Pose để lấy tư thế người và LSTM để phân loại **Fall** (dấu hiệu té ngã) / **ADL** (hoạt động bình thường). Có hai chức năng: phân tích video tải lên và webcam trực tiếp. Mô hình đã có sẵn, không cần huấn luyện lại.

## 1. Chuẩn bị

- Dùng **Windows 10/11 và Python 3.11 bản 64 bit** theo hướng dẫn này. Khi cài Python, chọn **Add python.exe to PATH**.
- Cài Git để clone, hoặc tải ZIP từ GitHub.
- Có Internet khi cài thư viện lần đầu; có webcam nếu muốn phân tích trực tiếp.
- Không bắt buộc có GPU. Ứng dụng tự chọn CUDA nếu PyTorch nhận GPU tương thích; còn lại dùng CPU.

Mở PowerShell và kiểm tra:

```powershell
python --version
git --version
```

## 2. Tải dự án

```powershell
git clone https://github.com/Thiet-do-doc/Du_an_ai_te_nga.git
cd Du_an_ai_te_nga
```

Nếu không dùng Git: chọn **Code → Download ZIP** trên GitHub, giải nén và mở PowerShell trong thư mục chứa `app.py` và `requirements.txt`.

## 3. Cài thư viện

Chạy lần lượt trong thư mục dự án:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Các lệnh dùng trực tiếp Python trong `.venv`, không cần kích hoạt môi trường hoặc đổi Execution Policy. Việc tải PyTorch có thể mất vài phút. Nếu có nhiều bản Python, dùng `py -3.11 -m venv .venv` ở bước tạo môi trường.

Kiểm tra thư viện và nạp mô hình:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "from predict import FallPredictor, DEVICE; FallPredictor(); print('Model OK | Device:', DEVICE)"
```

Lệnh thứ hai cần in `Model OK | Device: cpu` hoặc `Model OK | Device: cuda`. Đây là kiểm tra nạp mô hình, chưa kiểm chứng độ chính xác nhận diện.

## 4. Chạy ứng dụng

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Mở [http://localhost:8501](http://localhost:8501) nếu trình duyệt chưa tự mở. Giữ PowerShell đang chạy, nhấn `Ctrl+C` để tắt.

**Lần chạy sau:** mở PowerShell trong thư mục dự án và chạy lại lệnh trên. Không cần tạo môi trường hay cài lại thư viện.

### Menu chức năng

Menu dọc bên trái có bốn mục:

- **Giám sát trực tiếp:** nhập `0` cho webcam mặc định, hoặc địa chỉ camera IP, nhấn **Bắt đầu giám sát**. Khi phát hiện dấu hiệu ngã, cảnh báo đỏ xuất hiện phía trên hình ảnh. Từng đoạn camera được lưu cùng kết quả để xem lại. Nhấn **Dừng giám sát** khi xong. Chuyển sang mục khác cũng dừng camera.
- **Xem lại sự cố:** lọc khoảng ngày, cảnh báo hoặc bản ghi chưa kiểm tra; phát lại, tải video và đánh dấu đã kiểm tra.
- **Thống kê an toàn:** xem số đoạn đã phân tích, cảnh báo, cảnh báo chưa kiểm tra, dữ liệu chưa đủ; xem biểu đồ theo ngày và tải báo cáo CSV.
- **Tải video lên:** chọn MP4, AVI hoặc MOV, nhấn **Phân tích và lưu kết quả**. Video và kết quả được lưu để xem lại.

Webcam thuộc máy đang chạy Python, không phải webcam trình duyệt từ xa. Giám sát hoạt động khi trang đang mở, không phải dịch vụ chạy nền. Mỗi đoạn camera gồm 16 khung hình lấy theo tốc độ xử lý; khi phân tích, ứng dụng lấy mẫu thành 32 bước. Cảnh báo xuất hiện sau khi đoạn ghi hình được phân tích, không phải ngay lập tức ở từng khung hình. CPU chậm có thể làm tăng độ trễ. Chưa có gửi email/SMS.

Ứng dụng lấy mẫu 32 khung hình trên toàn video tải lên và trả một kết quả chung; nên dùng video ngắn. Một cảnh báo trong thống kê tương ứng một đoạn video, không phải một sự cố độc lập. Các đoạn liên tiếp có thể cùng ghi nhận một lần ngã.

Video và lịch sử được lưu trên máy chạy ứng dụng trong `data/` (SQLite: `data/history.db`), không đưa lên GitHub. Muốn chuyển lịch sử sang máy khác cần sao chép cả cơ sở dữ liệu lẫn video và cập nhật đường dẫn video nếu vị trí thư mục thay đổi. Các đoạn ghi hình được giữ lại và sẽ tăng dung lượng ổ đĩa theo thời gian.

## 5. Cấu trúc chính

```text
Du_an_ai_te_nga/
├── app.py                  # Giao diện Streamlit
├── predict.py              # Trích xuất tư thế và dự đoán video
├── webcam.py               # Phân tích webcam
├── model.py                # Kiến trúc và hàm nạp LSTM
├── requirements.txt        # Thư viện ứng dụng
├── models/
│   ├── best_lstm.pt         # Trọng số LSTM
│   └── yolo11n-pose.pt      # Trọng số YOLO-Pose
└── .streamlit/config.toml   # Cấu hình giao diện
```

Hai tệp `.pt` đã có trong GitHub; giữ đúng tên và vị trí. Không cần tải dữ liệu huấn luyện để chạy ứng dụng.

Kho còn có `main.py`, `src/`, `config/`, `tests/` và `docs/` từ bộ khung tích hợp. `python main.py --demo` chạy dữ liệu **giả lập**, không chạy mô hình nhận diện thật. Để chạy giao diện này, cài **requirements.txt** và dùng **streamlit run app.py**.

## 6. Khắc phục lỗi

| Lỗi | Cách xử lý |
| --- | --- |
| Không tìm thấy Python | Cài Python 3.11 64 bit, bật PATH và mở lại PowerShell; hoặc thử `py -3.11`. |
| Không tìm thấy Git | Cài Git và mở lại PowerShell, hoặc tải ZIP. |
| `No module named ...` | Cài lại `requirements.txt` bằng Python trong `.venv`, dùng đúng lệnh chạy ở bước 4. |
| Thiếu mô hình | Kiểm tra `models/best_lstm.pt` và `models/yolo11n-pose.pt`, tải lại từ kho nếu thiếu. |
| Không mở được webcam | Đóng Camera/Teams/Zoom, cho phép ứng dụng desktop truy cập Camera trong cài đặt quyền riêng tư Windows. Nếu có nhiều camera, thử đổi chỉ số `0` trong `cv2.VideoCapture` ở `app.py` thành `1`. |
| Không đọc được video | Thử MP4 dùng codec H.264 và kiểm tra tệp không bị hỏng. |
| Cổng 8501 đang được dùng | Thêm `--server.port 8502` vào lệnh chạy rồi mở `http://localhost:8502`. |
| Xử lý chậm, webcam chưa có kết quả | Thử video ngắn trước, đóng ứng dụng nặng và cải thiện ánh sáng. CPU chậm có thể chưa thu đủ khung hình trong cửa sổ thời gian của webcam. |

Nếu cài thư viện thất bại, kiểm tra Python 3.11 64 bit và Internet. Thư viện hiện chưa được khóa toàn bộ phiên bản. Khi báo lỗi, gửi nội dung lỗi cùng kết quả `python --version` và `.\.venv\Scripts\python.exe -m pip freeze` để đối chiếu môi trường.

## 7. Dữ liệu đã xử lý và video gốc

Kho Git có sẵn thư mục `data/processed/`, gồm các tệp metadata, danh sách chia tập `train.csv`, `val.csv`, `test.csv` và đặc trưng tư thế `.npy` trong `pose_features/`. Clone kho sẽ tải cả dữ liệu đã xử lý này. Có thể chạy `python view_npy.py` sau khi cài thư viện để xem đặc trưng của mẫu huấn luyện đầu tiên.

**Video gốc được lưu riêng, không đưa lên GitHub:** [Tải bộ video gốc trên Google Drive](https://drive.google.com/file/d/1tuyy3yE0Etby3pE-xpzCCM6shAxUUZ7H/view?usp=drive_link). Người chỉ chạy ứng dụng với mô hình có sẵn không cần tải bộ video huấn luyện.

Mở link, tải bộ dữ liệu về rồi giải nén. Nếu Drive yêu cầu quyền truy cập, chủ sở hữu cần bật quyền xem cho người có đường liên kết để người tải kho mã nguồn có thể lấy dữ liệu.

Khi nhận được bộ video, giải nén để có cấu trúc:

```text
data/raw/VideoDataset/
├── ADL/
└── Fall/
```

Các tệp CSV hiện có đường dẫn video từ môi trường tạo dữ liệu ban đầu, ví dụ `/content/Fall_Detection_Project/VideoDataset/...`. Nếu dùng CSV để đọc video trên máy khác, cần đổi tiền tố đường dẫn này thành vị trí thực tế của `data/raw/VideoDataset/` trên máy đó. Việc này không cần thiết khi chạy giao diện `app.py` để phân tích video mới hoặc webcam.

Thư mục `data/raw/` tiếp tục bị Git bỏ qua để tránh đưa bộ video dung lượng lớn vào kho mã nguồn.

## Giới hạn

Mô hình chọn người có khung bao lớn nhất, chưa theo dõi từng người riêng biệt. Kết quả cần người kiểm tra lại. Ứng dụng phục vụ nghiên cứu và trình diễn, không thay thế hệ thống cảnh báo an toàn đã được kiểm chứng.
