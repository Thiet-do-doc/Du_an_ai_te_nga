
import tempfile
from pathlib import Path

import cv2
import streamlit as st

from predict import DEVICE, FallPredictor
from webcam import WebcamAnalyzer


# ============================================
# 1. CẤU HÌNH TRANG
# ============================================

st.set_page_config(
    page_title="FallGuard AI | Phát hiện té ngã",
    page_icon="🩵",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================
# 2. THIẾT KẾ GIAO DIỆN TRẮNG - XANH NHẠT
# ============================================

st.markdown(
    """
    <style>
      :root {
        color-scheme: light;
      }

      .stApp {
        background: #f5fbff;
        color: #17324d;
      }

      [data-testid="stHeader"] {
        background: rgba(245, 251, 255, 0.92);
      }

      [data-testid="stSidebar"] {
        background: #e9f6ff;
        border-right: 1px solid #d8ebf8;
      }

      [data-testid="stSidebar"] * {
        color: #214562;
      }

      .block-container {
        padding-top: 1.7rem;
        padding-bottom: 3rem;
        max-width: 1240px;
      }

      h1, h2, h3, p, label {
        color: #17324d;
      }

      /* Tiêu đề chính */

      .hero {
        background: linear-gradient(
          110deg,
          #ffffff 0%,
          #e3f4ff 100%
        );
        border: 1px solid #cfe8fa;
        border-radius: 22px;
        padding: 26px 30px;
        margin-bottom: 20px;
        box-shadow: 0 6px 24px rgba(48, 111, 153, 0.06);
      }

      .hero .eyebrow {
        color: #258ac2;
        font-size: 0.86rem;
        letter-spacing: 0.1em;
        font-weight: 700;
      }

      .hero h1 {
        margin: 0.45rem 0;
        color: #16466b;
        font-size: clamp(1.6rem, 2.8vw, 2.35rem);
      }

      .hero p {
        margin: 0;
        color: #56768f;
        font-size: 0.98rem;
      }

      /* Tiêu đề từng chức năng */

      .section-title {
        font-weight: 750;
        font-size: 1.28rem;
        color: #16466b;
        margin: 0.5rem 0 1rem;
      }

      .hint {
        color: #648298;
        font-size: 0.92rem;
        margin: 0 1rem 1rem 0;
      }

      /* Ô thông tin */

      .info-tile {
        background: white;
        border: 1px solid #dbeefa;
        border-radius: 15px;
        padding: 14px 18px;
        min-height: 90px;
      }

      .info-tile strong {
        display: block;
        color: #174a70;
        font-size: 1.1rem;
        margin-top: 4px;
      }

      .info-tile small {
        color: #6b899e;
        font-size: 0.78rem;
      }

      /* Metric */

      div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d9ebf9;
        border-radius: 15px;
        padding: 12px 17px;
      }

      div[data-testid="stMetricLabel"] {
        color: #527994;
      }

      div[data-testid="stMetricValue"] {
        color: #164e77;
      }

      /* Tab */

      div[data-baseweb="tab-list"] {
        gap: 10px;
        background: #e8f5ff;
        padding: 8px;
        border-radius: 14px;
      }

      button[data-baseweb="tab"] {
        border-radius: 10px;
        color: #4c7797;
        font-weight: 650;
      }

      button[data-baseweb="tab"][aria-selected="true"] {
        background: white;
        color: #0875b0;
      }

      /* Nút bấm */

      div.stButton > button[kind="primary"] {
        background: #258ec6;
        color: white;
        border: 1px solid #258ec6;
        border-radius: 10px;
        font-weight: 700;
      }

      div.stButton > button[kind="primary"]:hover {
        background: #1875a8;
        border-color: #1875a8;
      }

      div.stButton > button[kind="secondary"] {
        background: #ffffff;
        color: #2877a4;
        border: 1px solid #c5e5f7;
        border-radius: 10px;
      }

      /* Vùng tải video */

      [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px solid #d2e9f8;
        border-radius: 15px;
        padding: 12px;
      }

      /* CẢNH BÁO MÀU ĐỎ */

      .alert-danger {
        background: #fff0f1;
        border: 2px solid #ed4c58;
        border-radius: 14px;
        padding: 17px 19px;
        color: #a91425;
        font-weight: 800;
        margin-top: 12px;
      }

      /* Trạng thái bình thường */

      .alert-safe {
        background: #edf9f7;
        border: 1px solid #bce6da;
        border-radius: 14px;
        padding: 15px 18px;
        color: #176655;
        font-weight: 700;
        margin-top: 12px;
      }

      /* Đang xác nhận */

      .alert-wait {
        background: #fff9ed;
        border: 1px solid #ffe0a6;
        border-radius: 14px;
        padding: 15px 18px;
        color: #91600b;
        font-weight: 700;
        margin-top: 12px;
      }

      .footer {
        font-size: 0.82rem;
        color: #7893a6;
        margin-top: 1.3rem;
      }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================
# 3. TẢI MÔ HÌNH
# ============================================

@st.cache_resource
def load_predictor():
    return FallPredictor()


# ============================================
# 4. HIỂN THỊ TRẠNG THÁI
# ============================================

def status_card(kind, text):
    styles = {
        "danger": "alert-danger",
        "safe": "alert-safe",
        "wait": "alert-wait"
    }

    st.markdown(
        f'<div class="{styles[kind]}">{text}</div>',
        unsafe_allow_html=True
    )


# ============================================
# 5. QUẢN LÝ WEBCAM
# ============================================

def close_camera():
    camera = st.session_state.get("camera")

    if camera is not None:
        camera.release()

    st.session_state.camera = None
    st.session_state.analyzer = None
    st.session_state.camera_active = False


for key, initial in {
    "camera_active": False,
    "camera": None,
    "analyzer": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = initial


# ============================================
# 6. HEADER - TIÊU ĐỀ CHÍNH
# ============================================

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">
        AI-POWERED SAFETY MONITORING
      </div>

      <h1>
        🩵 HỆ THỐNG AI PHÁT HIỆN TÉ NGÃ
      </h1>

      <p>
        Phân tích tư thế con người qua video và webcam
        bằng YOLO11n-Pose kết hợp LSTM.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================
# 7. SIDEBAR - THÔNG TIN HỆ THỐNG
# ============================================

with st.sidebar:
    st.markdown("## 🩵 FallGuard AI")

    st.caption("Bảng điều khiển hệ thống")

    st.divider()

    st.markdown("**Mô hình**")

    st.write("🎯 YOLO11n-Pose")
    st.write("🧠 LSTM – 2 tầng, hidden size 64")
    st.write("📐 Đầu vào – 32 × 51")
    st.write(f"💻 Thiết bị – {DEVICE}")

    st.divider()

    st.markdown("**Nguyên tắc dự đoán**")

    st.write("🟢 ADL: điểm Fall dưới 50%")
    st.write("🔴 Fall: điểm Fall từ 50%")

    st.caption(
        "Webcam yêu cầu 3 dự đoán Fall liên tiếp "
        "trước khi phát cảnh báo."
    )


# ============================================
# 8. CÁC THẺ THÔNG TIN
# ============================================

p1, p2, p3 = st.columns(3)

with p1:
    st.markdown(
        """
        <div class="info-tile">
            <small>CÔNG NGHỆ</small>
            <strong>YOLO-Pose + LSTM</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

with p2:
    st.markdown(
        """
        <div class="info-tile">
            <small>ĐẦU VÀO MÔ HÌNH</small>
            <strong>32 frame × 51 đặc trưng</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

with p3:
    st.markdown(
        """
        <div class="info-tile">
            <small>CHẾ ĐỘ</small>
            <strong>Video MP4 / Webcam</strong>
        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")


# ============================================
# 9. TẠO HAI TAB CHỨC NĂNG
# ============================================

tab_video, tab_camera = st.tabs([
    "📁 Phân tích video",
    "📷 Webcam trực tiếp"
])


# ============================================
# TAB 1 - PHÂN TÍCH VIDEO MP4
# ============================================

with tab_video:

    st.markdown(
        '<div class="section-title">'
        '📁 Phân tích video tải lên'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="hint">'
        'Tải video lên để nhận kết quả Fall / ADL. '
        'Video hiển thị ở kích thước gọn, căn giữa trang.'
        '</p>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Chọn video MP4, AVI hoặc MOV",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_file is not None:

        # Video nhỏ bằng 50% vùng hiển thị
        left, middle, right = st.columns([1, 2, 1])

        with middle:
            st.video(uploaded_file)

        # Nút phân tích căn giữa
        btn_l, btn_center, btn_r = st.columns(
            [2, 1.5, 2]
        )

        with btn_center:
            analyze = st.button(
                "🔍 Phân tích video",
                type="primary",
                use_container_width=True
            )

        if analyze:
            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(uploaded_file.name).suffix
                ) as tmp:

                    tmp.write(uploaded_file.getvalue())
                    temp_path = Path(tmp.name)

                with st.spinner(
                    "Đang trích xuất keypoint và dự đoán..."
                ):
                    result = (
                        load_predictor()
                        .predict_video(temp_path)
                    )

                st.markdown(
                    "#### 📊 Kết quả phân tích"
                )

                st.metric(
                    "Tỷ lệ frame phát hiện người",
                    f"{result['valid_frame_ratio'] * 100:.1f}%"
                )

                if result["fall_score"] is None:

                    status_card(
                        "wait",
                        "⚠️ Không đủ dữ liệu tư thế để dự đoán."
                    )

                else:
                    score = result["fall_score"]

                    st.metric(
                        "Điểm dự đoán Fall",
                        f"{score * 100:.2f}%"
                    )

                    if result["label"] == "Fall":

                        status_card(
                            "danger",
                            "🚨 CẢNH BÁO: "
                            "PHÁT HIỆN DẤU HIỆU TÉ NGÃ!"
                        )

                    else:

                        status_card(
                            "safe",
                            "✅ Không phát hiện té ngã trong video."
                        )

            except Exception as exc:

                st.error(
                    f"Không thể phân tích video: {exc}"
                )

            finally:

                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)


# ============================================
# TAB 2 - WEBCAM TRỰC TIẾP
# ============================================

with tab_camera:

    st.markdown(
        '<div class="section-title">'
        '📷 Giám sát webcam trực tiếp'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="hint">'
        'Camera trên laptop chạy ứng dụng sẽ được '
        'sử dụng để phân tích chuyển động theo thời gian.'
        '</p>',
        unsafe_allow_html=True
    )

    action_l, action_c, action_r = st.columns(
        [2, 1.4, 2]
    )

    with action_c:

        if not st.session_state.camera_active:

            if st.button(
                "▶ Bật webcam",
                type="primary",
                use_container_width=True
            ):

                cam = cv2.VideoCapture(
                    0,
                    cv2.CAP_DSHOW
                )

                if not cam.isOpened():

                    cam.release()

                    st.error(
                        "Không mở được webcam. "
                        "Kiểm tra quyền truy cập camera "
                        "và ứng dụng khác đang sử dụng."
                    )

                else:

                    cam.set(
                        cv2.CAP_PROP_FRAME_WIDTH,
                        640
                    )

                    cam.set(
                        cv2.CAP_PROP_FRAME_HEIGHT,
                        480
                    )

                    try:

                        analyzer = WebcamAnalyzer(
                            load_predictor()
                        )

                        st.session_state.camera = cam
                        st.session_state.analyzer = analyzer
                        st.session_state.camera_active = True

                        st.rerun()

                    except Exception:
                        cam.release()
                        raise

        else:

            if st.button(
                "■ Tắt webcam",
                use_container_width=True
            ):

                close_camera()
                st.rerun()


    # ========================================
    # HIỂN THỊ CAMERA THEO THỜI GIAN
    # ========================================

    @st.fragment(run_every=0.3)
    def show_camera():

        if not st.session_state.camera_active:

            st.info(
                "📷 Webcam đang tắt. "
                "Nhấn Bật webcam để bắt đầu."
            )
            return

        cam = st.session_state.camera
        analyzer = st.session_state.analyzer

        try:

            ok, frame = cam.read()

            if not ok:

                close_camera()

                st.error(
                    "Không đọc được khung hình. "
                    "Webcam đã được đóng."
                )
                return

            # YOLO-Pose + LSTM
            rgb_frame = analyzer.process_frame(frame)

            # Camera nhỏ, căn giữa
            left, mid, right = st.columns([1, 3, 1])

            with mid:

                st.image(
                    rgb_frame,
                    channels="RGB",
                    use_container_width=True
                )

            st.markdown(
                "#### 📊 Trạng thái nhận diện"
            )

            col_a, col_b = st.columns(2)

            with col_a:

                st.metric(
                    "Tỷ lệ frame phát hiện người",
                    f"{analyzer.valid_ratio * 100:.1f}%"
                )

            with col_b:

                if analyzer.fall_score is None:
                    score_str = "Chưa có"

                else:
                    score_str = (
                        f"{analyzer.fall_score * 100:.2f}%"
                    )

                st.metric(
                    "Điểm dự đoán Fall",
                    score_str
                )

            # =================================
            # HIỂN THỊ CẢNH BÁO
            # =================================

            if analyzer.fall_score is None:

                status_card(
                    "wait",
                    "⌛ " + analyzer.label
                )

            elif analyzer.alert_active:

                status_card(
                    "danger",
                    "🚨 CẢNH BÁO ĐỎ: "
                    "NGHI NGỜ TÉ NGÃ!"
                )

            elif analyzer.label == "Fall":

                status_card(
                    "wait",
                    "⚠️ Phát hiện dấu hiệu Fall "
                    "– đang xác nhận..."
                )

            else:

                status_card(
                    "safe",
                    "✅ TRẠNG THÁI BÌNH THƯỜNG (ADL)"
                )

        except Exception as exc:

            close_camera()

            st.error(
                f"Lỗi webcam: {exc}"
            )

    show_camera()


# ============================================
# 10. FOOTER
# ============================================

st.markdown(
    """
    <p class="footer">
      Ứng dụng phục vụ nghiên cứu và trình diễn;
      không thay thế hệ thống cảnh báo an toàn
      được kiểm chứng thực tế.
    </p>
    """,
    unsafe_allow_html=True
)
