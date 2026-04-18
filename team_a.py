import cv2
import mediapipe as mp
import zmq
import json
import time

# --- 1. ZMQ PUBLISHER SETUP ---
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://127.0.0.1:5555")

print("📡 Live Vision Node Starting...")
print("Waiting for camera...")

# --- 2. MEDIAPIPE SETUP ---
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1 # Use 0 for faster performance, 2 for better accuracy
)

# --- 3. CAMERA SETUP ---
cap = cv2.VideoCapture(0)

# Optional: Force a lower resolution for faster processing if you notice lag
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("🎥 Camera active! Broadcasting to MuJoCo...")

# FPS Tracking variables
pTime = 0

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # MediaPipe needs RGB images
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = pose.process(img_rgb)

        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark
            
            # Extract 99 features (33 landmarks * 3 coordinates)
            flat_landmarks = []
            for lm in landmarks:
                flat_landmarks.extend([lm.x, lm.y, lm.z])
            
            payload = {
                "landmarks": flat_landmarks,
                "timestamp": time.time()
            }
            
            # Broadcast to inference_final.py
            socket.send_string(f"VISION {json.dumps(payload)}")
            
            # Draw the skeleton
            mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # --- CALCULATE & DISPLAY FPS ---
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        # Draw FPS on the screen (Green text)
        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, 'BROADCASTING TO G1', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Show the camera feed
        cv2.imshow("Team A - Live Tracker", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopping vision node...")
    
finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    socket.close()
    context.term()