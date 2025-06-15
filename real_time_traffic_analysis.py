# real_time_traffic_analysis.py 
import cv2
import numpy as np
from ultralytics import YOLO
from motion_utils import init_motion_detector, filter_by_motion
from collections import Counter          ### NEW ###

# ——— Model & thresholds ———
best_model = YOLO('models/best.pt')
heavy_traffic_threshold = 10      # > this many vehicles → "Heavy" traffic

# ——— Calibration ———
known_distance_m = 10.0           # real-world length for scaling
known_length_px  = 200.0
meters_per_pixel = known_distance_m / known_length_px
px_to_kmh        = meters_per_pixel * 3.6

# ——— Perspective calibration (unchanged) ———
vertices1 = np.array([(465,350),(609,350),(510,630),(2,630)],  dtype=np.float32)
vertices2 = np.array([(678,350),(815,350),(1203,630),(743,630)],dtype=np.float32)
dst_size = (500, 500)
dst_pts  = np.float32([[0,0],[500,0],[500,500],[0,500]])
H1 = cv2.getPerspectiveTransform(vertices1, dst_pts)
H2 = cv2.getPerspectiveTransform(vertices2, dst_pts)

# ——— Region & text settings (unchanged) ———
x1, x2 = 325, 635
lane_threshold = 609
text_pos_left   = (10,  50)
text_pos_right  = (820, 50)
int_pos_left    = (10, 100)
int_pos_right   = (820, 100)
font, font_scale, font_color = cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255)
bg_color = (0,0,255)

# ——— NEW: simple HSV colour classifier without white ———
def classify_colour(bgr_roi):
    """
    Improved car color classification.
    Detects: black, red — more tolerant to lighting.
    """
    if bgr_roi.shape[0] < 10 or bgr_roi.shape[1] < 10:
        return "other"

    # Center crop to avoid background pixels
    h, w = bgr_roi.shape[:2]
    crop = bgr_roi[h//4:3*h//4, w//4:3*w//4]

    blur = cv2.GaussianBlur(crop, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    s, v = cv2.split(hsv)[1], cv2.split(hsv)[2]
    mean_s, mean_v = np.mean(s), np.mean(v)

    if mean_v < 60:
        return "black"
    else:
        # Optional: hue histogram for red
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        dominant_hue = np.argmax(hist)
        if (dominant_hue <= 10 or dominant_hue >= 170) and mean_s > 70:
            return "red"
        return "other"

# ——— Initialize detectors ———
backSub, op_kernel, cl_kernel = init_motion_detector()

# Video I/O (unchanged)
cap  = cv2.VideoCapture('sample_video.mp4')
fps  = cap.get(cv2.CAP_PROP_FPS) or 20.0
four = cv2.VideoWriter_fourcc(*'XVID')
out  = cv2.VideoWriter('processed_sample_video.avi', four, fps,
                       (int(cap.get(3)), int(cap.get(4))))

# First frame prep (unchanged)
ret, prev_frame = cap.read()
if not ret: raise RuntimeError("Could not read first frame")
prev_gray  = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_bird1 = cv2.warpPerspective(prev_gray, H1, dst_size)
prev_bird2 = cv2.warpPerspective(prev_gray, H2, dst_size)

while True:
    ret, frame = cap.read()
    if not ret: break

    # ---- 1. MOTION & OPTICAL FLOW (unchanged) -----------------------------
    fgMask = backSub.apply(frame)
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bird1  = cv2.warpPerspective(gray, H1, dst_size)
    bird2  = cv2.warpPerspective(gray, H2, dst_size)
    flow1  = cv2.calcOpticalFlowFarneback(prev_bird1, bird1, None,
                                          0.5,3,15,3,5,1.2,0)
    flow2  = cv2.calcOpticalFlowFarneback(prev_bird2, bird2, None,
                                          0.5,3,15,3,5,1.2,0)
    mag1,_ = cv2.cartToPolar(flow1[...,0], flow1[...,1])
    mag2,_ = cv2.cartToPolar(flow2[...,0], flow2[...,1])
    prev_bird1, prev_bird2 = bird1, bird2

    # ---- 2. YOLO DETECTION -----------------------------------------------
    det_frame          = frame.copy()
    det_frame[:x1,:]   = 0
    det_frame[x2: ,:]  = 0
    results            = best_model.predict(det_frame, imgsz=640, conf=0.4)
    proc               = results[0].plot(line_width=1)
    proc[:x1,:]        = frame[:x1,:]
    proc[x2: ,:]       = frame[x2: ,:]
    cv2.polylines(proc, [vertices1.astype(int)], True, (0,255,0), 2)
    cv2.polylines(proc, [vertices2.astype(int)], True, (255,0,0), 2)

    # ---- 3. Filter boxes by real motion ----------------------------------
    raw_boxes  = [(int(a),int(b),int(c),int(d))
                  for a,b,c,d in results[0].boxes.xyxy.cpu().numpy()]
    mov_boxes  = filter_by_motion(fgMask, raw_boxes, op_kernel, cl_kernel)

    # ---- 4. INIT colour counters  ### NEW ### -----------------------------
    colour_counter = Counter()          # {'black':1, 'red':3, 'other':2, ...}

    # ---- 5. SPEED + COLOUR per vehicle -----------------------------------
    for (xA,yA,xB,yB) in mov_boxes:
        cx, cy = (xA+xB)/2, (yA+yB)/2
        if cx < lane_threshold:
            mag, H = mag1, H1
        else:
            mag, H = mag2, H2

        pt = np.array([[[cx, cy]]], dtype=np.float32)
        bx, by = cv2.perspectiveTransform(pt, H)[0][0]
        bx, by = int(bx), int(by)
        if 0 <= bx < dst_size[0] and 0 <= by < dst_size[1]:
            speed_kmh = mag[by, bx] * fps * px_to_kmh
        else:
            speed_kmh = 0.0

        # --- COLOUR  ### NEW ### -----------------------------------------
        roi = frame[yA:yB, xA:xB]
        vehicle_colour = classify_colour(roi) if roi.size else "other"
        colour_counter[vehicle_colour] += 1

        # Draw bbox, speed & colour
        cv2.rectangle(proc, (xA,yA), (xB,yB), (0,255,0), 2)
        cv2.putText(proc, f"{speed_kmh:.1f} km/h", (xA, yA-10),
                    font, 0.5, (255,255,0), 1, cv2.LINE_AA)
        cv2.putText(proc, vehicle_colour, (xA, yB+15),
                    font, 0.5, (0,255,255), 1, cv2.LINE_AA)

    # ---- 6. TRAFFIC COUNT / INTENSITY ------------------------------------
    left_count  = sum(1 for b in mov_boxes if b[0] < lane_threshold)
    right_count = len(mov_boxes) - left_count
    left_int  = "Heavy" if left_count  > heavy_traffic_threshold else "Smooth"
    right_int = "Heavy" if right_count > heavy_traffic_threshold else "Smooth"

    def draw_text(pos, text):
        cv2.rectangle(proc, (pos[0]-10,pos[1]-25),
                      (pos[0]+460,pos[1]+10), bg_color, -1)
        cv2.putText(proc, text, pos, font, font_scale,
                    font_color, 2, cv2.LINE_AA)

    draw_text(text_pos_left,  f"Vehicles L: {left_count}")
    draw_text(int_pos_left,   f"Intensity : {left_int}")
    draw_text(text_pos_right, f"Vehicles R: {right_count}")
    draw_text(int_pos_right,  f"Intensity : {right_int}")

    # ---- 7. SHOW COLOUR STATS  ### NEW ### --------------------------------
    y0 = 150
    # Removed 'white' from stats display
    for colour in ("black","red"):
        txt = f"{colour.capitalize()}: {colour_counter[colour]}"
        cv2.putText(proc, txt, (10,y0), font, 0.6, (200,200,200), 1, cv2.LINE_AA)
        y0 += 25

    # ---- 8. DISPLAY & WRITE ---------------------------------------------
    cv2.imshow('Traffic Analysis: speed + colour', proc)
    out.write(proc)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

