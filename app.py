import uuid
from datetime import date
from pathlib import Path

import cv2
import imageio_ffmpeg
import pandas as pd
import streamlit as st

from storage import ROOT, records, save_result, mark_reviewed

st.set_page_config(page_title='An Tâm • Giám sát an toàn', page_icon='◉', layout='wide')
st.markdown('''<style>
.block-container {padding-top:2.2rem; max-width:1400px; padding-bottom:3rem}
h1 {font-size:2rem!important; letter-spacing:-.04em} h2 {font-size:1.4rem!important}
[data-testid="stMetric"] {background:white; border:1px solid #E2E8EF; border-radius:14px; padding:18px}
[data-testid="stSidebar"] {border-right:1px solid #E2E8EF}
.stButton button, .stDownloadButton button {border-radius:9px; min-height:42px}
[data-testid="stCaptionContainer"] {color:#536577}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def predictor():
    from predict import FallPredictor
    return FallPredictor()


def result_message(result):
    if result['fall_score'] is None:
        st.warning('Chưa đủ dữ liệu tư thế để kết luận. Hãy kiểm tra góc quay và ánh sáng.')
    elif result['label'] == 'Fall':
        st.error('Phát hiện dấu hiệu té ngã • Cần kiểm tra')
    else:
        st.success('Không phát hiện dấu hiệu té ngã trong đoạn đã phân tích.')
    a, b = st.columns(2)
    a.metric('Điểm dự đoán té ngã', '—' if result['fall_score'] is None else f"{result['fall_score']:.1%}")
    b.metric('Khung hình có dữ liệu người', f"{result['valid_frame_ratio']:.0%}")


def stop_camera():
    capture = st.session_state.pop('capture', None)
    if capture is not None:
        capture.release()
    st.session_state['running'] = False
    st.session_state.pop('frames', None)


with st.sidebar:
    st.title('◉ An Tâm')
    st.caption('HỆ THỐNG GIÁM SÁT AN TOÀN')
    page = st.radio('Chức năng', ['Giám sát trực tiếp', 'Xem lại sự cố', 'Báo cáo & thống kê', 'Tải video lên'], label_visibility='collapsed')
    st.divider()
    st.caption('Dữ liệu được lưu trên máy chạy ứng dụng.')
    st.caption('AI hỗ trợ phát hiện dấu hiệu té ngã. Cần người kiểm tra lại cảnh báo.')

if page != 'Giám sát trực tiếp' and st.session_state.get('running'):
    stop_camera()

st.title(page)

if page == 'Tải video lên':
    st.caption('Chọn video, phân tích và lưu kết quả để xem lại bất cứ lúc nào.')
    with st.container(border=True):
        uploaded = st.file_uploader('Video cần kiểm tra', type=['mp4', 'avi', 'mov'])
        if uploaded:
            st.video(uploaded)
            upload_key = (uploaded.name, uploaded.size)
            if st.session_state.get('upload_key') != upload_key:
                st.session_state.pop('upload_result', None)
                st.session_state['upload_key'] = upload_key
            if st.button('Phân tích và lưu kết quả', type='primary'):
                ROOT.mkdir(exist_ok=True)
                path = ROOT / (uuid.uuid4().hex + Path(uploaded.name).suffix.lower())
                try:
                    path.write_bytes(uploaded.getvalue())
                    with st.spinner('Đang phân tích video…'):
                        result = predictor().predict_video(path)
                        save_result(path, uploaded.name, result)
                    st.session_state['upload_result'] = result
                    st.success('Đã lưu video và kết quả vào lịch sử.')
                except Exception as exc:
                    path.unlink(missing_ok=True)
                    st.error(f'Không thể phân tích video: {exc}')
            if 'upload_result' in st.session_state:
                result_message(st.session_state['upload_result'])

elif page == 'Giám sát trực tiếp':
    st.caption('Kết nối camera để theo dõi. Mỗi đoạn 32 khung hình được phân tích liên tục.')
    with st.container(border=True):
        source = st.text_input('Nguồn camera', value='0', help='0 là webcam trên máy chạy ứng dụng. Có thể nhập địa chỉ RTSP hoặc HTTP của camera IP.', disabled=st.session_state.get('running', False))
        st.caption('Webcam là camera của máy chủ chạy Streamlit, không phải camera của trình duyệt từ xa.')
        left, right = st.columns(2)
        start = left.button('Bắt đầu giám sát', type='primary', disabled=st.session_state.get('running', False), use_container_width=True)
        if right.button('Dừng giám sát', disabled=not st.session_state.get('running', False), use_container_width=True):
            stop_camera()
        if start:
            try:
                with st.spinner('Đang kết nối và tải mô hình…'):
                    predictor()
                    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
                    if not cap.isOpened():
                        cap.release()
                        raise ValueError('Không kết nối được camera. Kiểm tra nguồn và quyền truy cập.')
                    st.session_state.update(capture=cap, running=True, frames=[], last_result=None)
            except Exception as exc:
                st.error(str(exc))

    @st.fragment(run_every=0.2 if st.session_state.get('running') else None)
    def live_view():
        if not st.session_state.get('running'):
            st.info('Camera đang dừng. Chọn nguồn rồi nhấn Bắt đầu giám sát.')
            return
        ok, frame = st.session_state.capture.read()
        if not ok:
            stop_camera()
            st.error('Mất kết nối camera. Nhấn Dừng giám sát hoặc tải lại trang để kết nối lại.')
            return
        st.success('Đang giám sát • Camera đã kết nối')
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.session_state.frames.append(frame)
        if len(st.session_state.frames) >= 32:
            frames = st.session_state.frames
            st.session_state.frames = []
            ROOT.mkdir(exist_ok=True)
            path = ROOT / (uuid.uuid4().hex + '.mp4')
            writer = imageio_ffmpeg.write_frames(str(path), (frame.shape[1], frame.shape[0]), fps=5, codec='libx264', pix_fmt_in='bgr24', macro_block_size=1)
            try:
                writer.send(None)
                for item in frames:
                    writer.send(item.tobytes())
                writer.close()
                result = predictor().predict_video(path)
                save_result(path, 'Camera trực tiếp', result)
                st.session_state.last_result = result
            except Exception as exc:
                writer.close()
                path.unlink(missing_ok=True)
                stop_camera()
                st.error(f'Giám sát đã dừng: {exc}')
        if st.session_state.get('last_result'):
            result_message(st.session_state.last_result)
        else:
            st.caption('Đang thu thập đủ 32 khung hình cho lần phân tích đầu tiên…')
    live_view()

else:
    rows = records()
    st.caption('Kết quả thực tế từ video tải lên và các đoạn camera đã phân tích.')
    if not rows:
        st.info('Chưa có dữ liệu. Hãy bắt đầu giám sát hoặc tải video lên để phân tích.')
    else:
        df = pd.DataFrame(rows)
        df['day'] = pd.to_datetime(df['created']).dt.date
        a, b = st.columns(2)
        begin = a.date_input('Từ ngày', value=df['day'].min())
        end = b.date_input('Đến ngày', value=date.today())
        if begin > end:
            st.warning('Ngày bắt đầu phải trước hoặc bằng ngày kết thúc.')
            st.stop()
        df = df[(df.day >= begin) & (df.day <= end)]
        if page == 'Xem lại sự cố':
            choice = st.selectbox('Hiển thị', ['Cảnh báo té ngã', 'Chưa đủ dữ liệu', 'Tất cả kết quả'])
            if choice == 'Cảnh báo té ngã':
                df = df[df.label == 'Fall']
            elif choice == 'Chưa đủ dữ liệu':
                df = df[df.score.isna()]
            if df.empty:
                st.info('Không có kết quả phù hợp với bộ lọc.')
            else:
                choices = {row['id']: row for row in df.to_dict('records')}
                selected = st.selectbox('Chọn bản ghi để xem lại', list(choices), format_func=lambda key: f"{choices[key]['created'].replace('T', ' ')} • {choices[key]['source']} • {'Đã xem' if choices[key]['reviewed'] else 'Chưa xem'}")
                row = choices[selected]
                with st.container(border=True):
                    video = Path(row['video'])
                    if video.is_file():
                        st.video(str(video))
                        st.download_button('Tải đoạn video', video.read_bytes(), file_name=video.name)
                    else:
                        st.warning('Tệp video không còn trên máy lưu trữ.')
                    result_message(dict(fall_score=row['score'], label=row['label'], valid_frame_ratio=row['coverage']))
                    if row['reviewed']:
                        st.success('Đã kiểm tra bản ghi này.')
                    elif st.button('Đánh dấu đã kiểm tra', type='primary'):
                        mark_reviewed(selected)
                        st.rerun()
        else:
            fall = df[df.label == 'Fall']
            a, b, c, d = st.columns(4)
            a.metric('Đoạn đã phân tích', len(df))
            b.metric('Cảnh báo té ngã', len(fall))
            c.metric('Cảnh báo chưa kiểm tra', int((fall.reviewed == 0).sum()))
            d.metric('Chưa đủ dữ liệu', int(df.score.isna().sum()))
            st.caption('Một cảnh báo tương ứng một đoạn video, không phải một sự cố hoặc một người riêng biệt.')
            if df.empty:
                st.info('Không có dữ liệu trong khoảng ngày đã chọn.')
            else:
                st.subheader('Cảnh báo theo ngày')
                counts = df.assign(alert=(df.label == 'Fall').astype(int)).groupby('day')['alert'].sum()
                st.bar_chart(counts, color='#16766B')
                report = df[['created', 'source', 'label', 'score', 'coverage', 'reviewed']].copy()
                report['label'] = report['label'].replace({'Fall': 'Cảnh báo té ngã', 'ADL': 'Không phát hiện té ngã'})
                report['reviewed'] = report['reviewed'].map({0: 'Chưa kiểm tra', 1: 'Đã kiểm tra'})
                report.columns = ['Thời gian', 'Nguồn', 'Kết quả', 'Điểm té ngã', 'Tỷ lệ khung hình có người', 'Trạng thái']
                st.subheader('Chi tiết phân tích')
                st.dataframe(report, hide_index=True, use_container_width=True)
                st.download_button('Tải báo cáo CSV', report.to_csv(index=False).encode('utf-8-sig'), 'bao-cao-an-toan.csv', 'text/csv')
