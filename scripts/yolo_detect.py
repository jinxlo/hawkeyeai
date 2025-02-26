from ultralytics import YOLO
import cv2
import mediapipe as mp
import time

# Load YOLO model
MODEL_PATH = "/home/ubuntu/hawkeye_ai/models/yolo11n.pt"
model = YOLO(MODEL_PATH)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# RTMP video stream URL (localhost since Nginx receives it on EC2)
video_stream = "rtmp://localhost/live/stream"

print("🔄 YOLO detection script is running, waiting for video stream...")

while True:
    cap = cv2.VideoCapture(video_stream)

    if cap.isOpened():
        print("✅ Video stream detected! Starting object detection...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Stream disconnected. Waiting for reconnection...")
                break  # Exit loop and retry

            # Run YOLO object detection
            results = model(frame)

            # Draw bounding boxes & labels
            for result in results:
                for obj in result.boxes.data:
                    x1, y1, x2, y2, conf, cls = obj
                    label = f"{model.names[int(cls)]} {conf:.2f}"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Run Pose Estimation
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb_frame)

            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display frame
            cv2.imshow("YOLO + Pose Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Press 'q' to exit

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("🚫 No stream available. Retrying in 5 seconds...")
        time.sleep(5)  # Wait before retrying

