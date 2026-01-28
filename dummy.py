from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


def generate_dummy_csv(
    out_path: str = "people_crossing_log.csv",
    days: int = 2,
    cams: int = 2,
    lines: int = 3,
    events_per_day: int = 300,
    seed: int = 42,
    tz: str = "Asia/Jakarta",
    overwrite: bool = True,
):
    """
    Generate dummy people crossing log with columns:
    timestamp (ISO with +07:00), camera, line, in, out
    """
    rng = random.Random(seed)
    out_fp = Path(out_path)

    # window waktu
    end = pd.Timestamp.now(tz=tz)
    start = end - pd.Timedelta(days=days)

    total_events = max(1, days * events_per_day)

    # random timestamps (uniform) dalam window
    ts = start + pd.to_timedelta(
        [rng.random() * (end - start).total_seconds() for _ in range(total_events)],
        unit="s",
    )

    cameras = [f"cam{i+1}" for i in range(cams)]
    line_ids = list(range(1, lines + 1))

    rows = []
    for t in ts:
        cam = rng.choice(cameras)
        line = rng.choice(line_ids)

        # event masuk/keluar (simple random)
        if rng.random() < 0.55:
            in_v, out_v = 1, 0
        else:
            in_v, out_v = 0, 1

        rows.append(
            {
                "timestamp": pd.Timestamp(t).isoformat(),  # contoh: 2026-01-14T09:00:00+07:00
                "camera": cam,
                "line": int(line),
                "in": int(in_v),
                "out": int(out_v),
            }
        )

    df = pd.DataFrame(rows).sort_values("timestamp")

    if overwrite or (not out_fp.exists()):
        df.to_csv(out_fp, index=False)
    else:
        df.to_csv(out_fp, mode="a", header=False, index=False)

    print(f"Dummy CSV created: {out_fp.resolve()} ({len(df)} rows)")


if __name__ == "__main__":
    generate_dummy_csv(
        out_path="people_crossing_log_test.csv",
        days=30,
        cams=1,
        lines=3,
        events_per_day=400,
        seed=123,
        overwrite=True,
    )
