"""
capture.py  –  Unitree G1 Edition
===================================
Two modes:

  live   – real-time webcam capture with interactive recording controls
  batch  – offline processing of an existing video file into a dataset

── LIVE MODE ─────────────────────────────────────────────────────────────────
  python capture.py live --out gesture_hands_up.h5 --label hands_up

  Controls
  ──────────────────────────────────────────────────
  SPACE       start / stop recording
  S           save current buffer immediately
  C           clear current buffer (start over)
  Q / ESC     quit (saves if frames > 0)

── BATCH MODE ─────────────────────────────────────────────────────────────────
  python capture.py batch --src recording.mp4 --out gesture_tpose.h5 --label t_pose
  python capture.py batch --src recording.mp4 --out out.npz --conf 0.6

  Processes every frame of the video automatically (no interaction needed).
  A progress bar is printed to the terminal.
  Optionally writes a side-by-side annotated video with --preview annotated.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from mp2mujoco import (
    MediaPipeToG1, DatasetRecorder,
    JOINT_NAMES, G1_JOINTS, DOF,
    G1Frame,
)


# ──────────────────────────────────────────────────────────────────────────────
# DRAWING / OVERLAY UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

_C = dict(
    red    = (0,   50,  255),
    green  = (0,  210,   80),
    cyan   = (0,  200,  200),
    yellow = (0,  200,  200),
    white  = (255,255,  255),
    dark   = ( 15, 15,   15),
    panel  = ( 28, 28,   38),
)

_FONT   = cv2.FONT_HERSHEY_SIMPLEX
_SMALL  = 0.38
_MED    = 0.52
_LARGE  = 0.68


def _put(img, text, xy, scale=_MED, color=_C["white"], thick=1):
    cv2.putText(img, text, xy, _FONT, scale, color, thick, cv2.LINE_AA)


def _draw_live_overlay(img, frame: G1Frame | None, recording: bool, n_frames: int):
    h, w = img.shape[:2]

    # ── Status bar ────────────────────────────────────────────────────────────
    cv2.rectangle(img, (0, 0), (w, 32), _C["dark"], -1)
    status = "● REC" if recording else "○ PAUSED"
    sc     = _C["red"] if recording else _C["cyan"]
    _put(img, status,                (10,  23), _LARGE, sc,        2)
    _put(img, f"Frames: {n_frames}", (150, 23), _MED,   _C["white"])

    if frame is None:
        _put(img, "No pose detected", (10, 70), _LARGE, _C["red"], 2)
        return

    conf_c = _C["green"] if frame.confidence > 0.65 else _C["red"]
    _put(img, f"Conf: {frame.confidence:.2f}", (w - 170, 23), _MED, conf_c)

    # ── Joint angle panel (right edge) ────────────────────────────────────────
    pw = 210
    x0 = w - pw - 4
    cv2.rectangle(img, (x0, 36), (w - 4, 36 + DOF * 16 + 8), _C["panel"], -1)

    for i, (jspec, angle) in enumerate(zip(G1_JOINTS, frame.angles)):
        t   = (angle - jspec.lo) / (jspec.hi - jspec.lo + 1e-9)
        col = _C["green"] if 0.05 < t < 0.95 else _C["red"]
        row = 36 + 8 + i * 16
        # mini bar
        bar_w = int(t * 50)
        cv2.rectangle(img, (x0 + 2, row - 10), (x0 + 52, row - 3), (50, 50, 60), -1)
        cv2.rectangle(img, (x0 + 2, row - 10), (x0 + 2 + bar_w, row - 3), col, -1)
        _put(img, f"{jspec.name[:14]:<14} {angle:+.2f}", (x0 + 56, row - 2), _SMALL, col)

    # ── Hint strip ────────────────────────────────────────────────────────────
    cv2.rectangle(img, (0, h - 24), (w, h), _C["dark"], -1)
    _put(img, "SPACE:rec/pause  S:save  C:clear  Q:quit", (8, h - 7), _SMALL, _C["white"])


def _draw_batch_overlay(img, frame: G1Frame | None, frame_idx: int, total: int):
    h, w = img.shape[:2]
    pct  = frame_idx / max(total, 1)

    # Progress bar
    bar_h = 6
    cv2.rectangle(img, (0, h - bar_h), (w, h), (40, 40, 40), -1)
    cv2.rectangle(img, (0, h - bar_h), (int(w * pct), h), _C["green"], -1)
    cv2.rectangle(img, (0, h - 26), (w, h - bar_h), _C["dark"], -1)
    _put(img, f"Frame {frame_idx}/{total}  ({pct*100:.1f}%)", (8, h - 10), _SMALL, _C["white"])

    if frame is not None:
        conf_c = _C["green"] if frame.confidence > 0.65 else _C["red"]
        _put(img, f"Conf: {frame.confidence:.2f}", (w - 160, h - 10), _SMALL, conf_c)


# ──────────────────────────────────────────────────────────────────────────────
# SHARED MEDIAPIPE SETUP
# ──────────────────────────────────────────────────────────────────────────────

def _make_pose(complexity: int = 1):
    return mp.solutions.pose.Pose(
        static_image_mode        = False,
        model_complexity         = complexity,
        smooth_landmarks         = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5,
    )


def _process_frame(pose, bgr, converter: MediaPipeToG1, ts: float):
    """Run pose on one BGR frame, return (annotated_bgr, G1Frame | None)."""
    rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    frame   = None

    if results.pose_world_landmarks:
        lms   = results.pose_world_landmarks.landmark
        frame = converter.convert(lms, timestamp=ts)
        if not converter.is_valid(frame):
            frame = None

    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            bgr,
            results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return bgr, frame


# ──────────────────────────────────────────────────────────────────────────────
# MODE A: LIVE CAPTURE
# ──────────────────────────────────────────────────────────────────────────────

def run_live(
    src:        int | str,
    out_path:   Path,
    label:      str,
    max_frames: int,
    conf_thresh: float,
    complexity:  int,
):
    converter = MediaPipeToG1(confidence_threshold=conf_thresh)
    recorder  = DatasetRecorder(max_frames=max_frames)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"[Error] Cannot open source: {src!r}")

    recording    = False
    last_frame:  G1Frame | None = None
    saved_once   = False

    print("\n[Live] Pipeline ready.")
    print("  SPACE  = start / stop recording")
    print("  S      = save buffer now")
    print("  C      = clear buffer")
    print("  Q/ESC  = quit\n")

    with _make_pose(complexity) as pose:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            bgr, g1frame = _process_frame(pose, bgr, converter, time.time())
            last_frame = g1frame

            if recording and g1frame is not None:
                if not recorder.record(g1frame):
                    print("[Live] Buffer full – auto-saving and stopping.")
                    recorder.save(out_path, label=label)
                    saved_once = True
                    recording  = False

            _draw_live_overlay(bgr, last_frame, recording, recorder.n_frames)
            cv2.imshow("MediaPipe → G1  [LIVE]", bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):       # Q or ESC
                break
            elif key == ord(" "):
                recording = not recording
                state = "Recording" if recording else "Paused"
                print(f"[Live] {state}  ({recorder.n_frames} frames buffered)")
            elif key == ord("s"):
                if recorder.n_frames > 0:
                    recorder.save(out_path, label=label)
                    saved_once = True
                else:
                    print("[Live] Nothing to save yet.")
            elif key == ord("c"):
                recorder.clear()
                print("[Live] Buffer cleared.")

    cap.release()
    cv2.destroyAllWindows()

    if recorder.n_frames > 0 and not saved_once:
        recorder.save(out_path, label=label)
    elif recorder.n_frames == 0:
        print("[Live] No frames recorded.")


# ──────────────────────────────────────────────────────────────────────────────
# MODE B: BATCH VIDEO PROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def run_batch(
    src:        str | Path,
    out_path:   Path,
    label:      str,
    conf_thresh: float,
    complexity:  int,
    preview_out: Path | None,
    skip_frames: int,
):
    """
    Process every frame (or every Nth frame) of a video file.
    Optionally write an annotated preview video.
    """
    src = str(src)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"[Error] Cannot open video: {src!r}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n[Batch] Source : {src}")
    print(f"        Frames : {total_frames}  ({fps:.1f} fps)")
    print(f"        Output : {out_path}")
    if skip_frames > 1:
        print(f"        Skipping every {skip_frames-1} frames")

    converter = MediaPipeToG1(confidence_threshold=conf_thresh)
    recorder  = DatasetRecorder(max_frames=total_frames + 1)

    # Optional preview video writer
    writer = None
    if preview_out is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(preview_out), fourcc, fps, (w, h))
        print(f"        Preview: {preview_out}")

    print()

    with _make_pose(complexity) as pose:
        idx       = 0
        recorded  = 0
        skipped   = 0
        low_conf  = 0
        no_pose   = 0

        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            idx += 1

            # Optional frame skip (reduces dataset size / speeds processing)
            if skip_frames > 1 and (idx % skip_frames) != 0:
                if writer is not None:
                    writer.write(bgr)
                continue

            ts  = idx / fps    # use video-time instead of wall-clock
            bgr, g1frame = _process_frame(pose, bgr, converter, ts)

            if g1frame is None:
                no_pose += 1
            elif not converter.is_valid(g1frame):
                low_conf += 1
            else:
                recorder.record(g1frame)
                recorded += 1

            _draw_batch_overlay(bgr, g1frame, idx, total_frames)

            if writer is not None:
                writer.write(bgr)

            # Terminal progress every 30 frames
            if idx % 30 == 0 or idx == total_frames:
                pct = idx / max(total_frames, 1) * 100
                bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
                print(f"\r  [{bar}] {pct:5.1f}%  recorded={recorded}", end="", flush=True)

    cap.release()
    if writer is not None:
        writer.release()

    print(f"\n\n[Batch] Done.")
    print(f"  Total frames processed : {idx}")
    print(f"  Recorded (good)        : {recorded}")
    print(f"  No pose detected       : {no_pose}")
    print(f"  Low confidence         : {low_conf}")

    if recorder.n_frames > 0:
        recorder.save(out_path, label=label)
    else:
        print("[Batch] No valid frames – dataset not saved.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MediaPipe → Unitree G1 angle capture",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # ── live ──────────────────────────────────────────────────────────────────
    L = sub.add_parser("live", help="Real-time webcam capture")
    L.add_argument("--src",        default=0,
                   help="Camera index (default 0)")
    L.add_argument("--out",        default="gesture.h5",
                   help="Output dataset path  (default: gesture.h5)")
    L.add_argument("--label",      default="",
                   help="Gesture label stored in metadata  (e.g. hands_up)")
    L.add_argument("--max-frames", default=15_000, type=int)
    L.add_argument("--conf",       default=0.5,    type=float,
                   help="Min landmark confidence  (default 0.5)")
    L.add_argument("--complexity", default=1,      type=int, choices=[0, 1, 2],
                   help="MediaPipe model complexity (0=fast, 2=accurate)")

    # ── batch ─────────────────────────────────────────────────────────────────
    B = sub.add_parser("batch", help="Offline processing of an existing video")
    B.add_argument("--src",        required=True,
                   help="Path to input video file")
    B.add_argument("--out",        default="gesture.h5",
                   help="Output dataset path  (default: gesture.h5)")
    B.add_argument("--label",      default="",
                   help="Gesture label stored in metadata")
    B.add_argument("--conf",       default=0.5,    type=float)
    B.add_argument("--complexity", default=1,      type=int, choices=[0, 1, 2])
    B.add_argument("--preview",    default=None,
                   help="Optional: path to write annotated output video")
    B.add_argument("--skip",       default=1,      type=int,
                   help="Process every Nth frame  (default 1 = every frame)")

    return p


def main():
    args = _build_parser().parse_args()

    if args.mode == "live":
        try:
            src = int(args.src)
        except (ValueError, TypeError):
            src = args.src
        run_live(
            src        = src,
            out_path   = Path(args.out),
            label      = args.label,
            max_frames = args.max_frames,
            conf_thresh= args.conf,
            complexity = args.complexity,
        )

    elif args.mode == "batch":
        run_batch(
            src        = args.src,
            out_path   = Path(args.out),
            label      = args.label,
            conf_thresh= args.conf,
            complexity = args.complexity,
            preview_out= Path(args.preview) if args.preview else None,
            skip_frames= args.skip,
        )


if __name__ == "__main__":
    main()