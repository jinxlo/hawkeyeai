from ultralytics import YOLO
import cv2
import time

# Path to the YOLO model (use absolute path)
MODEL_PATH = "/home/ubuntu/hawkeye_ai/models/yolo11n.pt"

# Load the YOLO model
model = YOLO(MODEL_PATH)

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

            # Display detected objects
            for result in results:
                print("📸 Objects detected:", result.names)  # Print detected objects

        cap.release()  # Release video stream when disconnected
    else:
        print("🚫 No stream available. Retrying in 5 seconds...")
        time.sleep(5)  # Wait before retrying

