# Fall Detection App

Ứng dụng Streamlit phát hiện dấu hiệu té ngã trong video bằng YOLO11n Pose và LSTM.

## Cài đặt và chạy

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Tải video MP4, AVI hoặc MOV lên giao diện, sau đó chọn **Phân tích video**.

## Cấu trúc

- `app.py`: giao diện Streamlit.
- `predict.py`: trích xuất tư thế và dự đoán từ 32 frame.
- `model.py`: định nghĩa và tải mô hình LSTM.
- `models/yolo11n-pose.pt`: trọng số YOLO Pose.
- `models/best_lstm.pt`: trọng số LSTM.

Ứng dụng tự chọn CUDA khi khả dụng, hoặc chạy trên CPU. Kết quả là dự đoán AI, chưa được xác nhận cho sử dụng an toàn thực tế.
