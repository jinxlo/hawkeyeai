from ultralytics import YOLO
import cv2
import mediapipe as mp
import time
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load YOLO model for object detection
MODEL_PATH = "/home/ubuntu/hawkeye_ai/models/yolo11n.pt"
model = YOLO(MODEL_PATH)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# RTMP video stream URL (localhost since Nginx receives it on EC2)
video_stream = "rtmp://localhost/live/stream"

# Logs storage
detection_logs = []


def detect_hand_position(landmarks):
    """
    Determines if hands are in pockets, raised, or by the sides.
    """
    if landmarks is None:
        return "No Pose Detected"

    # Get key points
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

    # Convert normalized coordinates to pixel space
    def get_coords(landmark):
        return (int(landmark.x * 640), int(landmark.y * 480))

    lw_x, lw_y = get_coords(left_wrist)
    rw_x, rw_y = get_coords(right_wrist)
    lh_x, lh_y = get_coords(left_hip)
    rh_x, rh_y = get_coords(right_hip)
    le_x, le_y = get_coords(left_elbow)
    re_x, re_y = get_coords(right_elbow)

    # Logic for determining hand positions
    if lw_y < le_y and rw_y < re_y:
        return "Hands Raised"
    elif (lh_y - 10 < lw_y < lh_y + 20) and (rh_y - 10 < rw_y < rh_y + 20):
        return "Hands in Pockets"
    elif lw_y > lh_y and rw_y > rh_y:
        return "Hands by Sides"
    else:
        return "Unknown Hand Position"


def generate_frames():
    global detection_logs
    cap = cv2.VideoCapture(video_stream)

    if not cap.isOpened():
        print("🚫 No stream available. Retrying...")
        return

    print("✅ Video stream detected! Starting object & pose detection...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Stream disconnected. Waiting for reconnection...")
            break

        # YOLO Object Detection
        results = model(frame)
        detected_objects = []

        for result in results:
            for obj in result.boxes.data:
                x1, y1, x2, y2, conf, cls = obj
                label = f"{model.names[int(cls)]} ({conf:.2f})"
                
                # Only log "person" detections
                if model.names[int(cls)] == "person":
                    detected_objects.append(label)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Pose Estimation
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_frame)

        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            hand_status = detect_hand_position(pose_results.pose_landmarks.landmark)
            detected_objects.append(hand_status)

        # Log detected objects & pose
        detection_logs.append(", ".join(detected_objects))
        detection_logs = detection_logs[-10:]  # Keep last 10 logs

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/logs')
def logs():
    return "\n".join(detection_logs)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

