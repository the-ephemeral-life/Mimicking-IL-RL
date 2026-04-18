"""
capture.py
==========
Real-time pipeline:  Webcam → MediaPipe → mp2mujoco → G1 angles → Dataset

Controls
--------
  SPACE  – start / stop recording
  S      – save dataset immediately
  Q      – quit

Run
---
  python capture.py                      # webcam, save as dataset.h5
  python capture.py --src 0 --out run1.h5
  python capture.py --src video.mp4 --out run1.npz --fmt npz
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from mp2mujoco import MediaPipeToG1, DatasetRecorder, JOINT_NAMES, DOF, G1Frame


# ──────────────────────────────────────────────────────────────────────────────
# Overlay helpers
# ──────────────────────────────────────────────────────────────────────────────

_RED    = (0,   50, 255)
_GREEN  = (0,  220,  80)
_CYAN   = (0,  200, 200)
_WHITE  = (255,255, 255)
_DARK   = ( 10,  10,  10)

def _bar(val: float, lo: float, hi: float, w: int = 60) -> str:
    t = (val - lo) / (hi - lo + 1e-9)
    filled = int(t * w)
    return "[" + "█" * filled + "░" * (w - filled) + "]"

def _draw_overlay(img, frame: G1Frame | None, recording: bool, n_frames: int):
    h, w = img.shape[:2]

    # Status bar
    status    = "● REC" if recording else "○ PAUSED"
    status_c  = _RED if recording else _CYAN
    cv2.rectangle(img, (0, 0), (w, 30), _DARK, -1)
    cv2.putText(img, status,                     (10, 22),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_c, 2)
    cv2.putText(img, f"Frames: {n_frames:5d}",   (130, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE,   1)

    if frame is None:
        cv2.putText(img, "No pose detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _RED, 2)
        return

    # Confidence
    conf_c = _GREEN if frame.confidence > 0.6 else _RED
    cv2.putText(img, f"Conf: {frame.confidence:.2f}", (w - 180, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, conf_c, 1)

    # Mini angle display (right column)
    x0, y0 = w - 220, 45
    cv2.rectangle(img, (x0 - 5, y0 - 5), (w - 5, y0 + 19 * 16 + 5), (30, 30, 30), -1)
    for i, (name, angle) in enumerate(zip(JOINT_NAMES, frame.angles)):
        from mp2mujoco import G1_JOINTS
        spec = G1_JOINTS[i]
        t    = (angle - spec.lo) / (spec.hi - spec.lo + 1e-9)
        col  = (_GREEN if 0.1 < t < 0.9 else _RED)
        cv2.putText(img, f"{name[:18]:<18} {angle:+.2f}",
                    (x0, y0 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

    # Hint
    cv2.putText(img, "SPACE: rec/pause  S: save  Q: quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Main capture loop
# ──────────────────────────────────────────────────────────────────────────────

def run(src, out_path: Path, fmt: str, max_frames: int, conf_thresh: float):
    mp_pose   = mp.solutions.pose
    mp_draw   = mp.solutions.drawing_utils

    converter = MediaPipeToG1(confidence_threshold=conf_thresh)
    recorder  = DatasetRecorder(max_frames=max_frames)

    recording  = False
    last_frame: G1Frame | None = None

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"[Error] Cannot open source: {src!r}")

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as pose:

        print("[Capture] Pipeline ready.")
        print("  SPACE = start/stop recording")
        print("  S     = save now")
        print("  Q     = quit\n")

        while True:
            ok, bgr = cap.read()
            if not ok:
                print("[Capture] Stream ended.")
                break

            rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_world_landmarks:
                lms   = results.pose_world_landmarks.landmark
                frame = converter.convert(lms, timestamp=time.time())

                if converter.is_valid(frame):
                    last_frame = frame
                    if recording:
                        if not recorder.record(frame):
                            print("[Capture] Buffer full – stopping recording.")
                            recording = False
                else:
                    last_frame = None

                # Draw skeleton on image
                mp_draw.draw_landmarks(
                    bgr,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )
            else:
                last_frame = None

            _draw_overlay(bgr, last_frame, recording, recorder.n_frames)
            cv2.imshow("MediaPipe → Unitree G1", bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                recording = not recording
                print(f"[Capture] {'Recording' if recording else 'Paused'}")
            elif key == ord("s"):
                if recorder.n_frames > 0:
                    recorder.save(out_path, fmt=fmt)

    cap.release()
    cv2.destroyAllWindows()

    if recorder.n_frames > 0:
        recorder.save(out_path, fmt=fmt)
    else:
        print("[Capture] No frames recorded.")


# ──────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--src",       default=0,            help="Camera index or video path")
    p.add_argument("--out",       default="dataset.h5", help="Output file path")
    p.add_argument("--fmt",       default="h5",         choices=["h5", "npz", "csv"])
    p.add_argument("--max-frames",default=30_000, type=int)
    p.add_argument("--conf",      default=0.5,   type=float, help="Min landmark confidence")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    try:
        src = int(args.src)
    except ValueError:
        src = args.src

    run(
        src        = src,
        out_path   = Path(args.out),
        fmt        = args.fmt,
        max_frames = args.max_frames,
        conf_thresh= args.conf,
    )