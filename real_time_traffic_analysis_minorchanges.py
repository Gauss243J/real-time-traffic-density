import cv2
import numpy as np
from ultralytics import YOLO

# Load fine-tuned YOLOv8 model
model = YOLO("models/best.pt")

# Detection threshold
CONFIDENCE_THRESHOLD = 0.4

# Vertical cropping to reduce false positives (sky, irrelevant zones)
x_top, x_bottom = 50, 720

# Define 4 lane zones (adjust coordinates to your video!)
lanes = {
    "Left Lane 1": np.array([
        (320, 400),
        (410, 400),
        (350, 720),
        (200, 720)
    ], dtype=np.int32),

    "Left Lane 2": np.array([
        (410, 400),
        (460, 400),
        (500, 720),
        (350, 720)
    ], dtype=np.int32),
"Right Lane 1": np.array([
        (460, 400),
        (530, 400),
        (650, 720),
        (500, 720)
    ], dtype=np.int32),

    "Right Lane 2": np.array([
    (530, 400),  
    (650, 400),  
    (800, 720),  
    (650, 720)   
], dtype=np.int32)
}

# Lane colors
lane_colors = {
    "Left Lane 1": (0, 255, 0),
    "Left Lane 2": (0, 200, 0),
    "Right Lane 1": (255, 0, 0),
    "Right Lane 2": (200, 0, 0)
}

# Input video path
cap = cv2.VideoCapture("Test Video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    masked_frame = frame.copy()
    masked_frame[:x_top, :] = 0
    masked_frame[x_bottom:, :] = 0

    results = model.predict(masked_frame, imgsz=640, conf=CONFIDENCE_THRESHOLD)
    annotated_frame = results[0].plot()
    annotated_frame[:x_top, :] = frame[:x_top, :]
    annotated_frame[x_bottom:, :] = frame[x_bottom:, :]

    # Draw each lane
    for lane_name, polygon in lanes.items():
        cv2.polylines(annotated_frame, [polygon], True, lane_colors[lane_name], 2)

    # Initialize per-lane vehicle counters
    lane_counts = {name: 0 for name in lanes.keys()}

    for box in results[0].boxes.xyxy:
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        point = (int(center_x), int(center_y))

        for lane_name, polygon in lanes.items():
            if cv2.pointPolygonTest(polygon, point, False) >= 0:
                lane_counts[lane_name] += 1
                break

    # Display traffic info per lane
    y_offset = 30
    for i, (lane_name, count) in enumerate(lane_counts.items()):
        status = "Heavy" if count > 4 else "Smooth"
        text = f"{lane_name}: {count} | {status}"

        # Red box with white text
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        cv2.rectangle(annotated_frame, (10, y_offset), (10 + text_size[0] + 10, y_offset + 40), (0, 0, 255), -1)
        cv2.putText(annotated_frame, text, (15, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        y_offset += 50

    # Display the frame
    cv2.imshow("4-Lane Traffic Monitor", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
