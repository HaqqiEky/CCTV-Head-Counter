import os
import time
import csv
import cv2
import threading
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timezone, timedelta

import numpy as np
from ultralytics import YOLO

# =========================
# CONFIG
# =========================

# RTSP
RTSP_URLS = [
    "rtsp://admin:adm12345678@10.20.25.25:554/Streaming/channels/801",
    # Tambah jika ada cam lain: "rtsp://...."
]

# Model YOLO
MODEL_WEIGHTS = "yolo12n.pt"
CONF_THRES = 0.20
IOU_THRES = 0.35

# Frame size
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Target FPS processing
TARGET_FPS = 30

# CSV logging
LOG_CSV_PATH = "people_crossing_log.csv"

# Asia/Jakarta (UTC+7) tz for timestamps
WIB = timezone(timedelta(hours=7))

# Garis-garis line crossing
LINES: List[Tuple[Tuple[int,int], Tuple[int,int]]] = [
    ((224, 338),  (342, 333)),
    ((555, 325),  (715, 320)),
    ((938, 314),  (1060, 316)),
]

CENTER_HIT_TOL = 4  # toleransi jarak ke garis (px)

# Camera thread
class CameraWorker(threading.Thread):
    def __init__(self, name: str, url: str, width: Optional[int], height: Optional[int]):
        super().__init__(daemon=True)
        self.name = name
        self.url = url
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.running = False
        self._lock = threading.Lock()

    def run(self):
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.running = True
        print(f"[{self.name}] Camera thread started.")
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.25)
                continue

            if self.width and self.height:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

            with self._lock:
                self.latest_frame = frame

            time.sleep(0.005)

        self._release()
        print(f"[{self.name}] Camera thread stopped.")

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


# YOLO helpers
_models: Dict[int, YOLO] = {}

def load_model_per_cam(cam_idx: int, weights_path: str) -> YOLO:
    global _models
    if cam_idx not in _models:
        print(f"[YOLO] Loading model for cam {cam_idx+1}...")
        model = YOLO(weights_path)
        model.fuse()
        _models[cam_idx] = model
    return _models[cam_idx]

def get_person_class_id(model: YOLO) -> int:
    if hasattr(model, "names") and isinstance(model.names, dict):
        for i, n in model.names.items():
            if str(n).lower() == "person":
                return int(i)
    return 0


def run_inference_track(model: YOLO, frame_bgr: np.ndarray,
                        conf: float, iou: float, person_id: int):
    results = model.track(
        frame_bgr,
        conf=conf,
        iou=iou,
        classes=[person_id],         
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )
    return results[0]

# Line-crossing state & math

cross_state: Dict[int, Dict] = {}
log_lock = threading.Lock()

def _signed_side_and_t(cx, cy, P1, P2):
    x, y = float(cx), float(cy)
    x1, y1 = map(float, P1)
    x2, y2 = map(float, P2)
    dx, dy = (x2 - x1), (y2 - y1)
    vx, vy = (x - x1), (y - y1)
    cross = dx * vy - dy * vx
    eps = 1e-6
    signed = 1 if cross > eps else (-1 if cross < -eps else 0)
    denom = dx*dx + dy*dy if (dx*dx + dy*dy) > 0 else 1.0
    t = (vx*dx + vy*dy) / denom
    return signed, t, (dx, dy)

def _ensure_state(cam_idx: int, n_lines: int):
    if cam_idx not in cross_state:
        cross_state[cam_idx] = {
            "prev_pt": {},
            "lines": [
                {"prev_side": {}, "up": 0, "down": 0, "net": 0}
                for _ in range(n_lines)
            ],
        }
    else:
        cur = cross_state[cam_idx]
        if len(cur["lines"]) != n_lines:
            extra = n_lines - len(cur["lines"])
            if extra > 0:
                cur["lines"].extend({"prev_side": {}, "up": 0, "down": 0, "net": 0} for _ in range(extra))
            else:
                cur["lines"] = cur["lines"][:n_lines]

def csv_header(path: str):
    header = ["timestamp", "camera", "line", "in", "out"]
    exists = os.path.exists(path)
    if not exists or os.path.getsize(path) == 0:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
        print(f"[CSV] Header written to {path}")

def log_crossing(camera_name: str, line_idx: int, direction: str):
    ts = datetime.now(WIB).isoformat(timespec="seconds")
    if direction == "in":
        row = [ts, camera_name, line_idx, 1, 0]
    else:
        row = [ts, camera_name, line_idx, 0, 1]

    with log_lock:
        with open(LOG_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    print(f"[LOG] {ts} cam={camera_name} line={line_idx} dir={direction}")

def annotate_and_count_crossing_multi(res, person_id: int, cam_idx: int, min_delta: float):
    _ensure_state(cam_idx, n_lines=len(LINES))
    state_cam = cross_state[cam_idx]

    boxes = getattr(res, "boxes", None)
    if boxes is not None and getattr(boxes, "xyxy", None) is not None:
        xyxy = boxes.xyxy.detach().cpu().numpy()
        cls  = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else None
        ids  = boxes.id.detach().cpu().numpy().astype(int) if boxes.id is not None else None

        if ids is not None and cls is not None:
            for bb, c, tid in zip(xyxy, cls, ids):
                if c != person_id:
                    continue

                x1, y1, x2, y2 = bb

                # bottom-center bbox
                cx = int((x1 + x2) / 2)
                cy = int(y2) - 2

                prev_pt = state_cam["prev_pt"].get(tid, None)
                state_cam["prev_pt"][tid] = (cx, cy)
                if prev_pt is not None:
                    move = float(np.hypot(cx - prev_pt[0], cy - prev_pt[1]))
                    if move >= float(min_delta):
                        for idx, (p1, p2) in enumerate(LINES):
                            line_state = state_cam["lines"][idx]
                            curr_side, t, (dx, dy) = _signed_side_and_t(cx, cy, p1, p2)
                            prev_side = line_state["prev_side"].get(tid, None)

                            line_state["prev_side"][tid] = curr_side

                            # Cek apakah titik berada di proyeksi garis dan dekat garis
                            x1f, y1f = map(float, p1)
                            x2f, y2f = map(float, p2)
                            dir_len = (dx*dx + dy*dy) ** 0.5 if (dx*dx + dy*dy) > 0 else 1.0
                            cross = dx * (float(cy) - y1f) - dy * (float(cx) - x1f)
                            dist_to_line = abs(cross) / dir_len if dir_len > 0 else 1e9

                            # crossing syarat: pernah di dua sisi, di segmen garis, dan tidak menempel
                            if prev_side is None:
                                continue
                            if not (0.0 <= t <= 1.0):
                                continue
                            if curr_side == 0 or prev_side == 0 or (curr_side == prev_side):
                                continue

                            # Tentukan arah (up/down) pakai normal
                            nx, ny = -dy, dx
                            mx, my = (cx - prev_pt[0], cy - prev_pt[1])
                            dot_norm = mx * nx + my * ny

                            # Dir definisi sama dengan kode Streamlit: dot_norm > 0 = "down"/in
                            if dot_norm > 0:
                                line_state["down"] += 1
                                line_state["net"]  += 1
                                log_crossing(camera_name=f"cam{cam_idx+1}", line_idx=idx + 1, direction="in")
                            else:
                                line_state["up"]   += 1
                                line_state["net"]  -= 1
                                log_crossing(camera_name=f"cam{cam_idx+1}", line_idx=idx + 1, direction="out")

    line_states = state_cam["lines"]
    return line_states

def annotate_and_count_crossing(res, person_id: int, cam_idx: int, min_delta: float = 5.0):
    line_states = annotate_and_count_crossing_multi(res, person_id, cam_idx, min_delta)
    up_total   = sum(ls["up"] for ls in line_states)
    down_total = sum(ls["down"] for ls in line_states)
    net_total  = sum(ls["net"] for ls in line_states)
    return up_total, down_total, net_total


# MAIN LOOP
print("[DAEMON] Starting YOLO line-crossing daemon...")
csv_header(LOG_CSV_PATH)

# Start camera workers
width = FRAME_WIDTH if FRAME_WIDTH > 0 else None
height = FRAME_HEIGHT if FRAME_HEIGHT > 0 else None

workers: List[CameraWorker] = []
for idx, url in enumerate(RTSP_URLS):
    if not url:
        continue
    w = CameraWorker(name=f"cam{idx+1}", url=url, width=width, height=height)
    w.start()
    workers.append(w)

    if not workers:
        print("[ERROR] No camera workers started. Check RTSP_URLS.")
        break

interval = 1.0 / max(1, TARGET_FPS)

try:
    while True:
        start_t = time.time()
        total_net = 0

        for cam_idx, w in enumerate(workers):
            frame = w.read_latest()
            if frame is None:
                continue

            model_cam = load_model_per_cam(cam_idx, MODEL_WEIGHTS)
            person_id = get_person_class_id(model_cam)

            res = run_inference_track(
                model_cam, frame,
                conf=CONF_THRES, iou=IOU_THRES,
                person_id=person_id
            )

            up, down, net = annotate_and_count_crossing(
                res, person_id=person_id, cam_idx=cam_idx, min_delta=5.0
            )
            total_net += net

        elapsed = time.time() - start_t
        if elapsed < interval:
            time.sleep(max(0.0, interval - elapsed))

except KeyboardInterrupt:
    print("\n[DAEMON] Stopped by user (Ctrl+C).")
finally:
    for w in workers:
        w.stop()
    time.sleep(0.2)
    print("[DAEMON] All camera workers stopped.")
