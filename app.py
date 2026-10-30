import tempfile
from pathlib import Path

import streamlit as st

from predict import FallPredictor, DEVICE


st.set_page_config(
    page_title="AI Fall Detection",
    page_icon="📹",
    layout="wide"
)

st.title("HỆ THỐNG AI PHÁT HIỆN TÉ NGÃ")
st.caption("YOLO-Pose + LSTM | Fall Detection System")

st.sidebar.header("Thông tin hệ thống")
st.sidebar.write("YOLO-Pose: YOLO11n Pose")
st.sidebar.write("LSTM: 2 tầng, 64 hidden units")
st.sidebar.write("Chuỗi dữ liệu: 32 frame")
st.sidebar.write(f"Thiết bị: {DEVICE}")


@st.cache_resource
def load_predictor():
    return FallPredictor()


uploaded_file = st.file_uploader(
    "Chọn video để kiểm tra",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Phân tích video", type="primary"):
        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(uploaded_file.name).suffix
            ) as temp:
                temp.write(uploaded_file.getvalue())
                temp_path = Path(temp.name)

            with st.spinner(
                "Đang trích xuất tư thế và phân tích..."
            ):
                predictor = load_predictor()
                result = predictor.predict_video(temp_path)

            st.subheader("Kết quả phân tích")

            st.write(
                "Tỷ lệ frame phát hiện người:",
                f"{result['valid_frame_ratio'] * 100:.2f}%"
            )

            if result["fall_score"] is None:
                st.warning(
                    "Không đủ dữ liệu tư thế để dự đoán đáng tin cậy."
                )
            else:
                score = result["fall_score"]

                st.metric(
                    "Điểm dự đoán Fall",
                    f"{score * 100:.2f}%"
                )

                if result["label"] == "Fall":
                    st.error(
                        "CẢNH BÁO: Phát hiện dấu hiệu té ngã!"
                    )
                else:
                    st.success(
                        "Không phát hiện té ngã trong video."
                    )

            st.caption(
                "Kết quả là dự đoán của mô hình AI, "
                "chưa được xác nhận cho sử dụng an toàn thực tế."
            )

        except Exception as e:
            st.error(f"Lỗi phân tích: {e}")

        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)