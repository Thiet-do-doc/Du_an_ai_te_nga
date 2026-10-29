# Hợp đồng dữ liệu v0.1 — cả nhóm phải dùng chung

## Manifest CSV (Khuyến)
Các cột: video_id,path,subject_id,event_id,activity,label,fall_start_s,fall_end_s,split.
label: 0=không té ngã; 1=có té ngã trong clip. split: train/val/test.
subject_id lấy từ dữ liệu gốc; không tự đoán mã người. Nếu không có, ghi rõ cách chia thay thế.
event_id gom các góc quay/bản sao cùng một lần thực hiện. Các nhóm phải không giao giữa các tập.
fall_start_s/fall_end_s để trống khi chưa có annotation; không ghi 0 để thay cho thiếu nhãn.
Không mặc định 2017 là số loại hoạt động: kiểm chứng README nguồn rồi thống kê file thực.

## Frame (Mạnh)
iter_frames(source, sample_fps) dự kiến yield (timestamp_seconds, frame_bgr).
frame_bgr: ảnh OpenCV H x W x 3; timestamp tăng theo thời gian của nguồn.
Video dùng thời gian video; webcam dùng thời gian monotonic tương đối từ đầu phiên.
Không lấy mẫu bằng số frame cố định khi FPS các video khác nhau.

## PoseSample (Thiết)
source_id định danh phiên nguồn; track_id giữ cùng người qua thời gian.
joints: 17 bộ (x,y,confidence), thứ tự COCO:
nose,left_eye,right_eye,left_ear,right_ear,left_shoulder,right_shoulder,
left_elbow,right_elbow,left_wrist,right_wrist,left_hip,right_hip,left_knee,
right_knee,left_ankle,right_ankle.

Đề xuất chuẩn hóa x=(x_pixel-cx_bbox)/max(w_bbox,epsilon),
y=(y_pixel-cy_bbox)/max(h_bbox,epsilon). Lưu bbox=(cx/W,cy/H,w/W,h/H)
để giữ vị trí/kích thước trong ảnh. Kiểm chứng đặc trưng này bằng validation.
Khớp thiếu: x=y=confidence=0 và mask=False; khớp hợp lệ mask=True.
Ngưỡng confidence phải lưu trong metadata. Không coi tọa độ 0 là cách duy nhất nhận biết thiếu khớp.
Không nội suy bằng dữ liệu tương lai trong pipeline online. Nếu nội suy offline,
phải chọn chính sách có thể tái hiện online hoặc ghi rõ khác biệt và đánh giá ảnh hưởng.

## Chuỗi và vector đầu vào (Thiết + Võ)
SequenceBuffer trả tuple[PoseSample,...] cho đúng một người, cùng nguồn.
Mỗi timestep đề xuất flatten 17*(x,y,confidence,mask) rồi nối 4 bbox = 72 đặc trưng.
LSTM đầu vào dự kiến [batch,30,72]. Random Forest dùng thống kê của cùng cửa sổ.
Hai bên phải chốt vectorizer dùng chung trước khi trích xuất toàn bộ dữ liệu.
Không ghép chuỗi qua hai video, hai người hoặc hai tập dữ liệu.
Scaler nếu có chỉ fit trên train; lưu lại để áp dụng cho validation/test và camera.

## Nhãn cửa sổ (Khuyến + Võ)
Nhãn clip khác nhãn từng cửa sổ. Quy định khoảng té ngã và mức chồng lấn
cần thiết để một cửa sổ được gán 1; ghi rõ trong báo cáo. Nếu chỉ có nhãn clip,
chọn chiến lược phân loại clip trước hoặc bổ sung annotation, không gán toàn bộ
cửa sổ đầu video té ngã thành 1 một cách mặc định.
Khi cửa sổ có phần hậu sự kiện, phải tính phần thời gian chờ đó vào độ trễ online.

## Prediction và AlertEvent (Võ + Thiện)
Prediction chứa source_id,track_id,timestamp,score trong [0,1]. Timestamp là cuối cửa sổ.
Score là điểm mô hình, chỉ gọi là xác suất khi có cơ sở hiệu chỉnh.
AlertManager xác nhận qua nhiều cửa sổ, có cooldown và reset khi gián đoạn.
Giá trị config là mốc thử, chưa được chọn từ dữ liệu thật.
Cooldown hiện chỉ giới hạn tần suất, chưa đảm bảo một cảnh ngã kéo dài chỉ có một event.
Thiện bổ sung state machine theo sự kiện (normal/suspected/alerted/recovered)
và test video dài trước nghiệm thu. Không reset cooldown tùy từng frame.

## Model bàn giao (Võ)
model artifact và metadata.json chứa: model_version, schema_version, input_order,
label_map, sample_fps, sequence_length, stride, normalization, confidence_threshold,
missing_joint_policy, pose_weights/version, scaler (nếu có), alert_threshold,
split_seed, training_config và kết quả validation. Không đưa nhãn thật vào input.
