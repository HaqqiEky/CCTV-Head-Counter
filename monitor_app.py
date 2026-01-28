import time
import threading
from pathlib import Path
from typing import Optional, List
from datetime import date

import cv2
import numpy as np
import pandas as pd
import streamlit as st

TZ = "Asia/Jakarta"

# =========================
# RTSP camera reader (threaded)
# =========================
class CameraWorker(threading.Thread):
    def __init__(self, name: str, url: str, width: Optional[int] = None, height: Optional[int] = None, fps: int = 15):
        super().__init__(daemon=True)
        self.name = name
        self.url = url
        self.width = width
        self.height = height
        self.fps = max(1, int(fps))
        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.running = False
        self._lock = threading.Lock()

    def run(self):
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.running = True
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.25)
                continue
            if self.width and self.height:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            with self._lock:
                self.latest_frame = frame
            time.sleep(0.001)

        self._release()

    def read_latest(self):
        with self._lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop(self):
        self.running = False

    def _release(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None


# =========================
# CSV helpers
# =========================
def load_csv(csv_path):
    fp = Path(csv_path)
    if not fp.exists():
        return pd.DataFrame(columns=["timestamp", "camera", "line", "in", "out"])

    try:
        df = pd.read_csv(fp)

        # normalize expected columns
        for col in ["timestamp", "camera", "line", "in", "out"]:
            if col not in df.columns:
                df[col] = np.nan

        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize(TZ)
        else:
            ts = ts.dt.tz_convert(TZ)
        df["timestamp"] = ts

        for col in ["line", "in", "out"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        df["camera"] = df["camera"].fillna("").astype(str)

        # drop invalid timestamp
        df = df.dropna(subset=["timestamp"])
        return df

    except Exception:
        return pd.DataFrame(columns=["timestamp", "camera", "line", "in", "out"])


def filter_by_date(df, date) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    start_dt = pd.Timestamp(date).tz_localize(TZ)
    end_dt = start_dt + pd.Timedelta(days=1)

    out = df[(df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)].copy()
    return out


def compute_peak_net(df):
    if df is None or df.empty:
        return None, None

    df2 = df.sort_values("timestamp", ascending=True).copy()
    df2["_delta"] = df2["in"].astype(float) - df2["out"].astype(float)
    df2["_net"] = df2["_delta"].cumsum()

    if df2.empty:
        return None, None

    idx = df2["_net"].idxmax()
    peak_person = int(df2.loc[idx, "_net"])
    peak_time = df2.loc[idx, "timestamp"]
    return peak_person, peak_time


# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="People Counter Monitor", layout="wide")
st.title("People Counter Monitor")


# =========================
# Session state init
# =========================
if "cam_workers" not in st.session_state:
    st.session_state.cam_workers: List[Optional[CameraWorker]] = []

if "df_today" not in st.session_state:
    st.session_state.df_today = pd.DataFrame(columns=["timestamp", "camera", "line", "in", "out"])

if "df_latest" not in st.session_state:
    st.session_state.df_latest = pd.DataFrame(columns=["timestamp", "camera", "line", "in", "out"])

if "total_in" not in st.session_state:
    st.session_state.total_in = 0
if "total_out" not in st.session_state:
    st.session_state.total_out = 0
if "net" not in st.session_state:
    st.session_state.net = 0

if "per_line" not in st.session_state:
    st.session_state.per_line = pd.DataFrame(columns=["in", "out"])

if "peak_person_today" not in st.session_state:
    st.session_state.peak_person_today = None
if "peak_time_today" not in st.session_state:
    st.session_state.peak_time_today = None

# report cache (load manual)
if "report_loaded" not in st.session_state:
    st.session_state.report_loaded = False
if "report_df_day" not in st.session_state:
    st.session_state.report_df_day = pd.DataFrame(columns=["timestamp", "camera", "line", "in", "out"])
if "report_peak_person" not in st.session_state:
    st.session_state.report_peak_person = None
if "report_peak_time" not in st.session_state:
    st.session_state.report_peak_time = None


def stop_workers():
    for w in st.session_state.cam_workers:
        if w:
            try:
                w.stop()
            except Exception:
                pass
    time.sleep(0.2)
    st.session_state.cam_workers = []


def start_workers(urls, width, height, fps):
    stop_workers()
    workers: List[Optional[CameraWorker]] = []
    for i, u in enumerate(urls):
        if not u.strip():
            workers.append(None)
            continue
        w = CameraWorker(name=f"cam{i+1}", url=u.strip(), width=width, height=height, fps=fps)
        w.start()
        workers.append(w)
    st.session_state.cam_workers = workers


# =========================
# Sidebar (Mode Split)
# =========================
with st.sidebar:
    st.subheader("Mode")
    mode = st.radio("Pilih mode", ["Realtime", "Report"], horizontal=True, key="mode_radio")

    st.divider()
    st.subheader("CSV")
    csv_path = st.text_input("CSV path", value="people_crossing_log.csv", key="csv_path")

    if mode == "Realtime":
        csv_refresh_sec = st.slider("CSV refresh (sec)", 1, 30, 3, 1, key="csv_refresh_sec")

        st.divider()
        st.subheader("CCTV Preview (no inference)")
        num_cams = st.number_input("Jumlah kamera", 1, 8, 1, 1, key="num_cams")

        default_urls = [
            "rtsp://admin:adm12345678@10.20.25.25:554/Streaming/channels/801",
            "rtsp://admin:adm12345678@10.20.25.25:554/Streaming/channels/802",
            "rtsp://admin:adm12345678@10.20.25.25:554/Streaming/channels/803",
            "rtsp://admin:adm12345678@10.20.25.25:554/Streaming/channels/804",
            "rtsp://admin:adm12345678@10.20.25.25:554/Streaming/channels/805",
            "rtsp://admin:adm12345678@10.20.25.25:554/Streaming/channels/806",
            "rtsp://admin:adm12345678@10.20.25.25:554/Streaming/channels/807",
            "rtsp://admin:adm12345678@10.20.25.25:554/Streaming/channels/808",
        ]

        urls: List[str] = []
        for i in range(num_cams):
            val = default_urls[i] if i < len(default_urls) else ""
            urls.append(st.text_input(f"RTSP Cam {i+1}", value=val, key=f"rtsp_{i}"))

        st.divider()
        st.subheader("Preview Options")
        preview_fps = st.slider("Target preview FPS", 1, 30, 10, 1, key="preview_fps")
        frame_width = st.number_input("Frame width (0=auto)", 0, 4096, 1280, 16, key="frame_width")
        frame_height = st.number_input("Frame height (0=auto)", 0, 4096, 720, 16, key="frame_height")

        st.divider()
        run_preview = st.toggle("Start Preview", value=False, help="Mulai/berhenti preview RTSP", key="run_preview")

    else:
        # report controls
        report_date = st.date_input(
            "Tanggal laporan",
            value=pd.Timestamp.now().date(),
            key="report_date_input",
        )
        load_report_btn = st.button("Load Report", type="primary", use_container_width=True)

if mode == "Report":
    stop_workers()


# =========================
# Layout
# =========================
left, right = st.columns([2, 1])

if mode == "Realtime":
    with left:
        st.subheader("Live CCTV")

    with right:
        st.subheader("Information")        
else:
    st.subheader("Report Viewer")
    right_panel_ph = st.empty()



# =========================
# Realtime UI
# =========================
if mode == "Realtime":
    with right:
        try:
            view_mode = st.segmented_control(
                "View",
                options=["Log (Latest)", "Aggregates", "Per Line (total)"],
                default="Log (Latest)",
                label_visibility="collapsed",
                key="realtime_view_mode",
            )
        except Exception:
            view_mode = st.radio(
                "View",
                ["Log (Latest)", "Aggregates", "Per Line (total)"],
                horizontal=True,
                label_visibility="collapsed",
                key="realtime_view_mode_fallback",
            )

        right_panel_ph = st.empty()

    # CCTV grid
    with left:
        grid_cols = st.columns(min(3, int(st.session_state.get("num_cams", 1))), gap="small")
        frame_slots = [grid_cols[i % len(grid_cols)].empty() for i in range(int(st.session_state.get("num_cams", 1)))]


# =========================
# REPORT MODE
# =========================
if mode == "Report":
    st.info("Mode Report: data tidak auto-refresh. Tekan **Load Report** untuk membaca CSV dan menampilkan laporan.")

    if "load_report_btn" in locals() and load_report_btn:
        df_all = load_csv(csv_path)
        df_day = filter_by_date(df_all, report_date)

        peak_p, peak_t = compute_peak_net(df_day)

        st.session_state.report_loaded = True
        st.session_state.report_df_day = df_day
        st.session_state.report_peak_person = peak_p
        st.session_state.report_peak_time = peak_t

    if not st.session_state.report_loaded:
        st.warning("Belum ada data report. Pilih tanggal lalu klik **Load Report**.")
    else:
        st.subheader("Report Summary")
        c1, c2 = st.columns(2)
        c1.metric("Peak Person", "-" if st.session_state.report_peak_person is None else st.session_state.report_peak_person)
        c2.metric(
            "Peak Time",
            "-" if st.session_state.report_peak_time is None else st.session_state.report_peak_time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        st.divider()
        st.subheader("Detail (df_day)")
        df_day_show = st.session_state.report_df_day.copy()
        if df_day_show is None or df_day_show.empty:
            st.info("Tidak ada data pada tanggal tersebut.")
        else:
            df_day_show = df_day_show.sort_values("timestamp", ascending=True)
            df_day_show["tanggal"] = df_day_show["timestamp"].dt.strftime("%Y-%m-%d")
            df_day_show["jam"] = df_day_show["timestamp"].dt.strftime("%H:%M:%S")

            ordered_cols = ["tanggal", "jam", "camera", "line", "in", "out"]
            cols = [c for c in ordered_cols if c in df_day_show.columns]
            st.dataframe(df_day_show[cols], width="stretch", height=520, hide_index=True)

            csv_bytes = df_day_show[cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download laporan (CSV)",
                data=csv_bytes,
                file_name=f"people_inout_{report_date}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # stop here
    st.stop()


# =========================
# REALTIME MODE
# =========================
# start/stop workers based on toggle
if mode == "Realtime":
    width = int(frame_width) if int(frame_width) > 0 else None
    height = int(frame_height) if int(frame_height) > 0 else None

    if run_preview and not st.session_state.cam_workers:
        start_workers(urls=urls, width=width, height=height, fps=int(preview_fps))
    if (not run_preview) and st.session_state.cam_workers:
        stop_workers()

    # main loop realtime
    last_csv_time = 0.0
    try:
        while run_preview and st.session_state.cam_workers:
            start_t = time.time()

            # 1) Update frames
            for i, w in enumerate(st.session_state.cam_workers):
                if i >= len(frame_slots):
                    break

                if w is None:
                    frame_slots[i].warning(f"Cam {i+1}: RTSP kosong")
                    continue

                frame = w.read_latest()
                if frame is None:
                    frame_slots[i].warning(f"Cam {i+1}: menunggu frame…")
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_slots[i].image(rgb, caption=f"Camera {i+1}", width="stretch")

            # 2) Refresh CSV
            now = time.time()
            if now - last_csv_time >= float(csv_refresh_sec):
                df_all = load_csv(csv_path)

                # filter TODAY
                today_date = pd.Timestamp.now().date()
                df_today = filter_by_date(df_all, today_date)

                st.session_state.df_today = df_today.copy()

                if not df_today.empty:
                    st.session_state.df_latest = df_today.sort_values("timestamp", ascending=False).head(200)
                else:
                    st.session_state.df_latest = pd.DataFrame(columns=["timestamp", "camera", "line", "in", "out"])

                st.session_state.total_in = int(df_today["in"].sum()) if not df_today.empty else 0
                st.session_state.total_out = int(df_today["out"].sum()) if not df_today.empty else 0
                st.session_state.net = st.session_state.total_in - st.session_state.total_out

                if not df_today.empty:
                    st.session_state.per_line = df_today.groupby("line")[["in", "out"]].sum().astype(int).sort_index()
                else:
                    st.session_state.per_line = pd.DataFrame(columns=["in", "out"])

                peak_p, peak_t = compute_peak_net(df_today)
                st.session_state.peak_person_today = peak_p
                st.session_state.peak_time_today = peak_t

                last_csv_time = now

            # 3) Render panel
            right_panel_ph.empty()
            with right_panel_ph.container():
                if view_mode == "Log (Latest)":
                    st.subheader("Log (Latest) - Today")
                    st.dataframe(st.session_state.df_latest, width="stretch", hide_index=True)

                elif view_mode == "Aggregates":
                    st.subheader("Aggregates - Today")
                    a1, a2, a3 = st.columns(3)
                    a1.metric("Total In", st.session_state.total_in)
                    a2.metric("Total Out", st.session_state.total_out)
                    a3.metric("People on This Floor", st.session_state.net)

                    st.divider()
                    b1, b2 = st.columns(2)
                    b1.metric("Peak Person (Today)", "-" if st.session_state.peak_person_today is None else st.session_state.peak_person_today)
                    
                    pt = st.session_state.peak_time_today
                    if pt is None:
                        b2.metric("Peak Time (Today)", "-")
                    else:
                        b2.metric("Peak Time (Today)", pt.strftime("%H:%M:%S"))

                else:
                    st.subheader("Per Line (total) - Today")
                    st.dataframe(st.session_state.per_line, width="stretch", hide_index=True)

            # 4) Throttle sesuai FPS preview
            elapsed = time.time() - start_t
            target_interval = 1.0 / max(1, int(preview_fps))
            if elapsed < target_interval:
                time.sleep(max(0.0, target_interval - elapsed))

    finally:
        if not run_preview:
            stop_workers()
