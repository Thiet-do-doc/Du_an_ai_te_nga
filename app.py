import sqlite3
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from streamlit.components.v1 import declare_component

from camera_feed import CameraFeed

from storage import ROOT, records, save_result, mark_reviewed, mark_many_reviewed, delete_record
from live_processing import analyze_clip, analyze_browser_clip

st.set_page_config(page_title='An Tâm • Giám sát an toàn', page_icon='🛡️', layout='wide', initial_sidebar_state='expanded')
st.markdown("""<style>
:root {color-scheme:light}
.stApp {background:#f3f3f2;color:#272b30}
[data-testid="stHeader"] {background:rgba(243,243,242,.95)}
.block-container {padding-top:2.5rem;padding-bottom:3rem;max-width:1280px}
h1 {font-size:2rem!important;letter-spacing:-.035em;font-weight:750!important}
h2,h3 {color:#30343a;letter-spacing:-.02em}
[data-testid="stSidebar"] {background:#e6e7e8;border-right:1px solid #d2d4d6;min-width:285px}
[data-testid="stSidebar"] .block-container {padding-top:2rem}
[data-testid="stSidebar"] [data-testid="stRadio"],[data-testid="stSidebar"] [data-testid="stRadio"] div:has([role="radiogroup"]),[data-testid="stSidebar"] [role="radiogroup"] {width:100%!important}
[data-testid="stSidebar"] [role="radiogroup"] {gap:6px;align-items:stretch!important}
[data-testid="stSidebar"] [role="radiogroup"] label {display:flex!important;background:transparent;border:1px solid transparent;padding:13px 16px;border-radius:12px;width:100%!important;box-sizing:border-box;cursor:pointer;transition:background .15s}
[data-testid="stSidebar"] [role="radiogroup"] label>div>div:first-child {display:none}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background:#eef0f1}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#fff;border-color:#b7bec4;box-shadow:0 2px 7px #242b3210}
[data-testid="stSidebar"] [role="radiogroup"] label p {font-size:1rem;font-weight:600;color:#30363c}
[data-testid="stMetric"] {background:#fff;border:1px solid #dfe1e3;border-radius:16px;padding:20px}
[data-testid="stMetricLabel"] {color:#565d65}
[data-testid="stMetricValue"] {color:#272d33;font-size:2rem}
[data-testid="stVerticalBlockBorderWrapper"]>div {border-radius:16px!important}
.stButton button,.stDownloadButton button {border-radius:10px;min-height:46px;font-weight:600}
[data-testid="stCaptionContainer"] {color:#626970;line-height:1.7}
[data-testid="stFileUploader"] {background:#fff;border:1px dashed #bac0c6;border-radius:14px;padding:14px}
.live-alert {background:#fff0f0;border:2px solid #d74343;border-left:8px solid #c53535;border-radius:14px;padding:22px 24px;margin:16px 0;color:#932020;box-shadow:0 4px 18px #9a202010}
.live-alert strong {display:block;font-size:1.5rem;margin-bottom:6px}
.camera-alert {background:#b4232e;color:#fff;border:2px solid #941c26;box-shadow:0 4px 22px #b4232e30}
.stApp:has(.camera-alert) {background:#fff1f1}
[data-testid="stMain"]:has(.camera-alert) {box-shadow:inset 0 0 0 5px #c53535}
.insight {margin:6px 0 14px;background:#fff;border:1px solid #dfe1e3;border-radius:14px;padding:14px 18px;height:100%}
.insight span {display:block;color:#626970;font-size:.82rem}
.insight b {display:block;color:#272d33;font-size:1.25rem;margin-top:2px}
.risk {display:inline-block;padding:3px 12px;border-radius:999px;font-size:.85rem;font-weight:700}
.risk-high {background:#fde8e8;color:#9b1c1c}.risk-mid {background:#fff3d6;color:#8a5a00}.risk-low {background:#e4f3e8;color:#1f6b3a}.risk-none {background:#eceeef;color:#575e65}
.page-kicker {color:#747b82;font-size:.78rem;font-weight:700;letter-spacing:.12em;margin-bottom:8px}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def predictor():
    from predict import FallPredictor
    return FallPredictor()


def result_message(result):
    if result['fall_score'] is None:
        st.warning('Chưa đủ dữ liệu tư thế để kết luận. Hãy kiểm tra góc quay và ánh sáng.')
    elif result['label'] == 'Fall':
        st.markdown('<div class="live-alert" role="alert"><strong>⚠ CẢNH BÁO TÉ NGÃ</strong>Phát hiện dấu hiệu té ngã. Hãy kiểm tra ngay người trong khu vực giám sát.</div>', unsafe_allow_html=True)
    elif result['label'] == 'Lying':
        st.warning('⚠ Phát hiện người vừa chuyển sang tư thế nằm. Hãy kiểm tra xem đó là nghỉ ngơi hay té ngã.')
    else:
        st.success('Không phát hiện dấu hiệu té ngã trong đoạn đã phân tích.')
    if result['fall_score'] is not None:
        st.caption('Kết quả hỗ trợ kiểm tra an toàn; hãy xác nhận tình trạng thực tế khi có cảnh báo.')


def risk_badge(score):
    if score is None or pd.isna(score):
        return '<span class="risk risk-none">Chưa đủ dữ liệu</span>'
    if score >= .9:
        return '<span class="risk risk-high">Nguy cơ rất cao</span>'
    if score >= .7:
        return '<span class="risk risk-high">Nguy cơ cao</span>'
    if score >= .5:
        return '<span class="risk risk-mid">Cần kiểm tra</span>'
    return '<span class="risk risk-low">Bình thường</span>'


def result_details(score, coverage):
    if score is None or pd.isna(score):
        st.metric('Độ phủ tư thế', f'{coverage:.0%}')
        return
    st.markdown(risk_badge(score), unsafe_allow_html=True)
    a, b = st.columns(2)
    a.metric('Điểm nguy cơ té ngã', f'{score:.0%}')
    b.metric('Độ phủ tư thế', f'{coverage:.0%}')


CAMERA_RETRY_SECONDS = 5
CONFIRM_CLIPS = 1  # số đoạn liên tiếp cùng ra té ngã mới phát cảnh báo
camera_preview = declare_component('camera_preview', path=str(Path(__file__).parent / 'camera_preview'))


browser_camera = declare_component('browser_camera', path=str(Path(__file__).parent / 'browser_camera'))


def stop_camera():
    capture = st.session_state.pop('capture', None)
    if capture is not None:
        capture.release()
    executor = st.session_state.pop('analysis_executor', None)
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)
    st.session_state.pop('analysis_future', None)
    st.session_state['running'] = False
    st.session_state.pop('frames', None)
    st.session_state.pop('frame_times', None)
    st.session_state.pop('alert_until', None)
    for key in ('last_result', 'camera_session', 'result_session', 'analysis_session',
                'last_preview_id', 'analysis_error', 'native_webcam', 'last_browser_clip',
                'session_clips', 'session_alerts', 'fall_streak'):
        st.session_state.pop(key, None)


def compact_video(video):
    # Center media in a column occupying 45% of the available content width.
    _, media, _ = st.columns([0.275, 0.45, 0.275])
    with media:
        st.video(video)


with st.sidebar:
    st.title('🛡️ An Tâm')
    st.caption('An toàn mỗi ngày, an tâm mỗi lúc.')
    st.divider()
    navigation = {'Giám sát trực tiếp': '📹', 'Xem lại sự cố': '🎞️', 'Thống kê an toàn': '📊', 'Tải video lên': '📤'}
    page = st.radio('Chức năng', list(navigation), format_func=lambda name: f'{navigation[name]}  {name}', label_visibility='collapsed')
    st.divider()
    st.caption('Dữ liệu được lưu trên máy chạy ứng dụng.')
    st.caption('Hãy kiểm tra tình trạng thực tế khi có cảnh báo.')

if page != 'Giám sát trực tiếp' and st.session_state.get('running'):
    stop_camera()

st.markdown('<div class="page-kicker">KHÔNG GIAN AN TOÀN CỦA BẠN</div>', unsafe_allow_html=True)
st.title(page)

if page == 'Tải video lên':
    st.caption('Chọn video, phân tích và lưu kết quả để xem lại bất cứ lúc nào.')
    with st.container(border=True):
        uploaded = st.file_uploader('Video cần kiểm tra', type=['mp4', 'avi', 'mov'])
        if uploaded:
            compact_video(uploaded)
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
                shown = st.session_state['upload_result']
                result_message(shown)
                result_details(shown['fall_score'], shown['valid_frame_ratio'])

elif page == 'Giám sát trực tiếp':
    if not st.session_state.get('running'):
        stop_camera()
    st.caption('Theo dõi hình ảnh trực tiếp và nhận cảnh báo trên màn hình khi có dấu hiệu té ngã.')
    with st.container(border=True):
        source = st.text_input('Nguồn camera', value='0', help='0 là webcam trên máy chạy ứng dụng. Có thể nhập địa chỉ RTSP hoặc HTTP của camera IP.', disabled=st.session_state.get('running', False))
        host = st.context.headers.get('Host', '').split(':')[0].lower() if hasattr(st, 'context') else ''
        local_webcam = source.strip() == '0' and host in ('localhost', '127.0.0.1')
        if local_webcam:
            st.caption('Webcam sẽ hiển thị ngay trong trình duyệt. Hãy cho phép sử dụng camera khi được hỏi.')
        else:
            st.caption('Nguồn hình ảnh được mở trên máy chạy ứng dụng.')
        left, right = st.columns(2)
        start = left.button('Bắt đầu giám sát', type='primary', disabled=st.session_state.get('running', False), use_container_width=True)
        if right.button('Dừng giám sát', disabled=not st.session_state.get('running', False), use_container_width=True):
            stop_camera()
            st.rerun()
        if start:
            st.session_state.pop('camera_error', None)
            try:
                with st.spinner('Đang chuẩn bị giám sát…'):
                    predictor()
                    feed = None
                    if not local_webcam:
                        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
                        if not cap.isOpened():
                            cap.release()
                            raise ValueError('Không kết nối được camera. Kiểm tra nguồn và quyền truy cập.')
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        if source.isdigit():
                            cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        feed = CameraFeed(cap)
                    st.session_state.update(capture=feed, native_webcam=local_webcam, last_browser_clip=0, last_preview_id=0, running=True, frames=[], frame_times=[], last_result=None, alert_until=0, next_sample_at=0, camera_session=uuid.uuid4().hex, result_session=None, analysis_session=None, analysis_future=None, analysis_executor=ThreadPoolExecutor(max_workers=1, thread_name_prefix='fall-analysis'))
                    st.rerun()
            except Exception as exc:
                st.error(str(exc))

    refresh = 0.5 if st.session_state.get('native_webcam') else 0.1

    @st.fragment(run_every=refresh if st.session_state.get('running') else None)
    def live_view():
        if not st.session_state.get('running'):
            if error := st.session_state.pop('camera_error', None):
                st.error(error)
                return
            st.info('Camera đang dừng. Chọn nguồn rồi nhấn Bắt đầu giám sát.')
            return
        native = st.session_state.native_webcam
        if not native:
            frame, captured_at, packets = st.session_state.capture.snapshot(st.session_state.last_preview_id)
            if packets:
                st.session_state.last_preview_id = packets[-1][0]
            age = time.monotonic() - captured_at
            if age >= CAMERA_RETRY_SECONDS:
                stop_camera()
                st.session_state.camera_error = 'Camera không gửi hình trong 5 giây. Kiểm tra kết nối rồi nhấn Bắt đầu giám sát để thử lại.'
                st.rerun()
            if frame is None:
                st.info('Đang chờ hình ảnh đầu tiên từ camera…')
            elif age > 0.5:
                st.warning('Camera đang gián đoạn. Đang hiển thị hình cuối cùng nhận được.')
            else:
                st.success('Đang giám sát • Camera đã kết nối')
        _, media, _ = st.columns([0.275, 0.45, 0.275])
        alert_area = st.container()
        pending = st.session_state.get('analysis_future')
        if pending is not None and pending.done():
            st.session_state.analysis_future = None
            try:
                if st.session_state.get('analysis_session') == st.session_state.get('camera_session'):
                    result = pending.result()
                    st.session_state.pop('analysis_error', None)
                    st.session_state.last_result = result
                    st.session_state.result_session = st.session_state.camera_session
                    detected = (result['label'] == 'Fall'
                                and result['fall_score'] is not None
                                and result['fall_score'] >= 0.9
                                and result['valid_frame_ratio'] >= 0.8)
                    streak = st.session_state.get('fall_streak', 0) + 1 if detected else 0
                    st.session_state.fall_streak = streak
                    confirmed = streak >= CONFIRM_CLIPS
                    st.session_state.alert_until = time.monotonic() + 10 if confirmed else 0
                    st.session_state.session_clips = st.session_state.get('session_clips', 0) + 1
                    st.session_state.session_alerts = st.session_state.get('session_alerts', 0) + int(confirmed)
            except Exception as exc:
                st.session_state.last_result = None
                st.session_state.alert_until = 0
                st.session_state.analysis_error = f'Lỗi phân tích đoạn vừa ghi: {exc}. Hệ thống sẽ thử lại ở đoạn tiếp theo.'
        # Keep only a rolling window of fresh samples while inference is busy.
        # Preview never waits for encoding/inference and old clips cannot queue up.
        if not native and packets and captured_at >= st.session_state.next_sample_at:
            st.session_state.frames.append(frame)
            st.session_state.frame_times.append(captured_at)
            st.session_state.frames = st.session_state.frames[-16:]
            st.session_state.frame_times = st.session_state.frame_times[-16:]
            st.session_state.next_sample_at = captured_at + 0.2
        if not native and st.session_state.get('analysis_future') is None and len(st.session_state.frames) >= 16:
            frames = st.session_state.frames
            timestamps = st.session_state.frame_times
            st.session_state.frames = frames[-8:]
            st.session_state.frame_times = timestamps[-8:]
            st.session_state.analysis_future = st.session_state.analysis_executor.submit(
                analyze_clip, frames, timestamps, predictor(), ROOT)
            st.session_state.analysis_session = st.session_state.camera_session
        current_result = (st.session_state.get('last_result')
                          if st.session_state.get('result_session') == st.session_state.get('camera_session') else None)
        alert_active = (current_result is not None and current_result['label'] == 'Fall'
                        and time.monotonic() < st.session_state.get('alert_until', 0))
        if st.session_state.get('analysis_error'):
            alert_area.warning(st.session_state.analysis_error)
        elif alert_active:
            with alert_area:
                st.markdown('<div class="live-alert camera-alert" role="alert" aria-live="assertive"><strong>⚠ CẢNH BÁO TÉ NGÃ</strong>Hãy kiểm tra ngay người trong khu vực camera đang giám sát.</div>', unsafe_allow_html=True)
        elif current_result and current_result['label'] == 'Fall' and st.session_state.get('fall_streak', 0) == 1:
            alert_area.info('Phát hiện dấu hiệu bất thường, đang xác nhận ở đoạn ghi hình tiếp theo…')
        elif current_result and current_result['label'] != 'Fall':
            with alert_area:
                result_message(current_result)
        else:
            st.caption('Đang quan sát chuyển động. Kết quả sẽ xuất hiện sau đoạn ghi hình đầu tiên…')
        if st.session_state.get('session_clips'):
            last = st.session_state.get('last_result')
            if last is None:
                detail = ''
            elif last['fall_score'] is None:
                detail = ' • Đoạn gần nhất: chưa đủ dữ liệu (cần thấy cả người từ đầu đến chân)'
            else:
                detail = f" • Đoạn gần nhất: điểm té ngã {last['fall_score']:.0%}"
            st.caption(f"Phiên này: {st.session_state.session_clips} đoạn đã phân tích • {st.session_state.get('session_alerts', 0)} cảnh báo{detail}")
        with media:
            if native:
                clip = browser_camera(alert=alert_active, key=st.session_state.camera_session)
                if clip and clip.get('error'):
                    stop_camera()
                    st.session_state.camera_error = clip['error']
                    st.rerun()
                if (clip and clip.get('sequence', 0) > st.session_state.last_browser_clip
                        and st.session_state.analysis_future is None):
                    st.session_state.last_browser_clip = clip['sequence']
                    st.session_state.analysis_future = st.session_state.analysis_executor.submit(
                        analyze_browser_clip, clip['samples'], predictor(), ROOT)
                    st.session_state.analysis_session = st.session_state.camera_session
            else:
                camera_preview(frames=packets, alert=alert_active, key=st.session_state.camera_session)
    live_view()

else:
    if page == 'Xem lại sự cố' and st.session_state.pop('video_deleted', False):
        st.success('Đã xóa bản ghi video khỏi lịch sử.')
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
            pending = st.checkbox('Chỉ xem bản ghi chưa kiểm tra')
            if pending:
                df = df[df.reviewed == 0]
            choice = st.selectbox('Hiển thị', ['Cảnh báo té ngã', 'Nằm xuống (cần kiểm tra)', 'Chưa đủ dữ liệu', 'Tất cả kết quả'])
            if choice == 'Cảnh báo té ngã':
                df = df[df.label == 'Fall']
            elif choice == 'Nằm xuống (cần kiểm tra)':
                df = df[df.label == 'Lying']
            elif choice == 'Chưa đủ dữ liệu':
                df = df[df.score.isna()]
            order = st.radio('Sắp xếp', ['Mới nhất', 'Điểm nguy cơ cao nhất'], horizontal=True)
            if order == 'Điểm nguy cơ cao nhất':
                df = df.sort_values('score', ascending=False, na_position='last')
            if df.empty:
                st.info('Không có kết quả phù hợp với bộ lọc.')
            else:
                waiting = df[df.reviewed == 0]
                st.caption(f'{len(df)} bản ghi • {len(waiting)} chưa kiểm tra')
                if len(waiting) > 1 and st.button(f'Đánh dấu {len(waiting)} bản ghi đang lọc là đã kiểm tra'):
                    mark_many_reviewed(list(waiting['id']))
                    st.rerun()
                choices = {row['id']: row for row in df.to_dict('records')}
                selected = st.selectbox('Chọn bản ghi để xem lại', list(choices), format_func=lambda key: f"{choices[key]['created'].replace('T', ' ')} • {choices[key]['source']} • {'Đã xem' if choices[key]['reviewed'] else 'Chưa xem'}")
                row = choices[selected]
                with st.container(border=True):
                    video = Path(row['video'])
                    if video.is_file():
                        compact_video(str(video))
                        st.download_button('Tải đoạn video', video.read_bytes(), file_name=video.name)
                    else:
                        st.warning('Tệp video không còn trên máy lưu trữ.')
                    result_message(dict(fall_score=row['score'], label=row['label'], valid_frame_ratio=row['coverage']))
                    result_details(row['score'], row['coverage'])
                    if row['reviewed']:
                        st.success('Đã kiểm tra bản ghi này.')
                    elif st.button('Đánh dấu đã kiểm tra', type='primary'):
                        mark_reviewed(selected)
                        st.rerun()
                    with st.popover('Xóa video'):
                        st.write('Xóa video đã chọn và kết quả phân tích khỏi lịch sử?')
                        if st.button('Xác nhận xóa', key=f'delete_{selected}'):
                            try:
                                delete_record(selected)
                            except (OSError, sqlite3.Error) as exc:
                                st.error(f'Chưa xóa được video: {exc}')
                            else:
                                st.session_state.video_deleted = True
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
                checked = len(fall) - int((fall.reviewed == 0).sum())
                fall_hours = pd.to_datetime(fall['created']).dt.hour
                peak = f'{int(fall_hours.mode()[0]):02d}:00' if len(fall) else '—'
                c1, c2, c3 = st.columns(3)
                c1.markdown(f'<div class="insight"><span>Đã kiểm tra</span><b>{checked / len(fall):.0%}</b></div>' if len(fall) else '<div class="insight"><span>Đã kiểm tra</span><b>—</b></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="insight"><span>Điểm nguy cơ cao nhất</span><b>{fall.score.max():.0%}</b></div>' if len(fall) else '<div class="insight"><span>Điểm nguy cơ cao nhất</span><b>—</b></div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="insight"><span>Khung giờ nhiều cảnh báo nhất</span><b>{peak}</b></div>', unsafe_allow_html=True)
                st.subheader('Cảnh báo theo ngày')
                counts = df.assign(alert=(df.label == 'Fall').astype(int)).groupby('day')['alert'].sum()
                st.bar_chart(counts, color='#687e8c')
                left, right = st.columns(2)
                with left:
                    st.subheader('Theo giờ trong ngày')
                    by_hour = fall_hours.value_counts().reindex(range(24), fill_value=0).sort_index()
                    st.bar_chart(by_hour, color='#b4232e')
                with right:
                    st.subheader('Theo nguồn')
                    by_source = fall.groupby('source').size().sort_values(ascending=False)
                    if by_source.empty:
                        st.caption('Chưa có cảnh báo té ngã trong khoảng ngày đã chọn.')
                    else:
                        st.bar_chart(by_source, color='#687e8c')
                report = df[['created', 'source', 'label', 'score', 'reviewed']].copy()
                report['created'] = report['created'].str.replace('T', ' ')
                report['label'] = report['label'].replace({'Fall': 'Cảnh báo té ngã', 'Lying': 'Nằm xuống – cần kiểm tra', 'ADL': 'Không phát hiện té ngã'})
                report['score'] = report['score'].map(lambda v: '—' if pd.isna(v) else f'{v:.0%}')
                report['reviewed'] = report['reviewed'].map({0: 'Chưa kiểm tra', 1: 'Đã kiểm tra'})
                report.columns = ['Thời gian', 'Nguồn', 'Kết quả', 'Điểm nguy cơ', 'Trạng thái']
                st.subheader('Chi tiết phân tích')
                st.dataframe(report, hide_index=True, use_container_width=True)
                st.download_button('Tải báo cáo CSV', report.to_csv(index=False).encode('utf-8-sig'), 'bao-cao-an-toan.csv', 'text/csv')
