"""
team_a.py  –  Vision Node
==========================
Captures pose from webcam via MediaPipe, broadcasts 99-float world-landmark
vectors over ZMQ PUB/SUB so inference_new.py can do IL arm control.

Key detail: we use pose_WORLD_landmarks (metric, hip-centred), NOT
pose_landmarks (image-space, normalised).  The IL brain was trained on
world coordinates, so this must stay consistent.
"""

import cv2
import mediapipe as mp
import zmq
import json
import time

# ── ZMQ Publisher ────────────────────────────────────────────────────────────
context = zmq.Context()
socket  = context.socket(zmq.PUB)
socket.bind("tcp://127.0.0.1:5555")

print("📡 Vision Node starting …")
print("   ZMQ publisher bound on tcp://127.0.0.1:5555")

# ── MediaPipe Pose ────────────────────────────────────────────────────────────
mp_pose  = mp.solutions.pose
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode        = False,
    model_complexity         = 1,        # 0=fast, 2=accurate
    smooth_landmarks         = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence  = 0.5,
)

# ── Camera ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("❌ Could not open webcam.")

print("🎥 Camera active — broadcasting world landmarks to inference node …\n")

prev_time   = time.time()
frames_sent = 0

try:
    while True:
        ok, img = cap.read()
        if not ok:
            print("⚠️  Empty camera frame — skipping.")
            continue

        # MediaPipe requires RGB
        rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        pose_detected = False

        if results.pose_world_landmarks:
            lms = results.pose_world_landmarks.landmark

            # 33 landmarks × 3 coords = 99 floats (world-space, metric, hip-centred)
            flat = []
            for lm in lms:
                flat.extend([lm.x, lm.y, lm.z])

            payload = json.dumps({"landmarks": flat, "timestamp": time.time()})
            socket.send_string(f"VISION {payload}")
            frames_sent += 1
            pose_detected = True

            # Draw skeleton on the BGR image for visual feedback
            mp_draw.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_style.get_default_pose_landmarks_style(),
            )

        # ── HUD ─────────────────────────────────────────────────────────────
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        status_txt  = "BROADCASTING" if pose_detected else "NO POSE"
        status_col  = (0, 255, 0)     if pose_detected else (0, 60, 255)

        cv2.putText(img, f"FPS: {fps:.0f}",      (16, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),   2)
        cv2.putText(img, status_txt,              (16, 76),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_col,    2)
        cv2.putText(img, f"Sent: {frames_sent}",  (16, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55,(200,200,200), 1)

        cv2.imshow("Team A – Vision Node", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\n⛔  Interrupted — shutting down.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    socket.close()
    context.term()
    print(f"✅ Vision node stopped. Total frames sent: {frames_sent}")