from flask import Flask, Response, render_template
from ultralytics import YOLO
import cv2

# Initialize Flask app
app = Flask(__name__)

# Path to the YOLO model
MODEL_PATH = "/home/ubuntu/hawkeye_ai/models/yolo11n.pt"

# Load YOLO model
model = YOLO(MODEL_PATH)

# RTMP video stream URL (localhost since Nginx receives it on EC2)
video_stream = "rtmp://localhost/live/stream"

def generate_frames():
    cap = cv2.VideoCapture(video_stream)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO object detection
        results = model(frame)

        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box[:4])
                confidence = float(conf) * 100  # Convert to percentage
                class_id = int(cls)
                label = model.names[class_id]  # Get label from class ID

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Display label & confidence
                label_text = f"{label}: {confidence:.1f}%"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

