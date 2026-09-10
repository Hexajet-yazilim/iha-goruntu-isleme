import cv2 as cv
import numpy as np
import time
import csv
from ultralytics import YOLO

# =============================================================================
# 🔹 BURAK: CSV DOSYA HAZIRLIĞI
# =============================================================================
log_filename = "flight_log.csv"
with open(log_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Zaman", "Durum", "Hedef_X", "Hedef_Y", "CMD_Roll", "CMD_Pitch", "FPS"])

# =============================================================================
# 🔹 İLKER: MODEL VE KAMERA BAŞLATMA
# =============================================================================
model = YOLO("yolov8n.pt")
cap = cv.VideoCapture(0)

# =============================================================================
# 🔹 TAHA: ZAMANLAMA VE PID DEĞİŞKENLERİ
# =============================================================================
prev_time = 0
last_log_time = 0
LOG_INTERVAL = 0.5  # Her 0.5 saniyede bir CSV'ye yaz (diski boğmamak için)

Kp = 0.1
MAX_SPEED = 100

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv.resize(frame, (640, 480))
    h, w, _ = frame_resized.shape
    center_screen_x, center_screen_y = w // 2, h // 2

    # =========================================================================
    # 1. BÖLÜM: İLKER'İN KODU (YOLOv8 Nesne Tespiti)
    # =========================================================================
    results = model(frame_resized, stream=True, conf=0.5, verbose=False)

    target_x, target_y, target_w, target_h = None, None, None, None
    center_x, center_y = None, None

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Hedef sınıf (örnek: person)
            if label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                target_w = x2 - x1
                target_h = y2 - y1
                target_x = x1
                target_y = y1
                center_x = target_x + target_w // 2
                center_y = target_y + target_h // 2

                cv.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(frame_resized, f"{label} {float(box.conf[0]):.2f}",
                           (x1, y1 - 8), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                break

    # =========================================================================
    # 2. BÖLÜM: TAHA'NIN KODU (FPS, Gecikme Ölçümü ve PID)
    # =========================================================================
    current_time = time.time()
    time_diff = current_time - prev_time
    prev_time = current_time

    fps = 1.0 / time_diff if time_diff > 0 else 0
    latency_ms = time_diff * 1000

    cmd_roll, cmd_pitch = 0, 0
    state = "LOST"

    if target_x is not None:
        state = "LOCKED"
        dx = center_x - center_screen_x
        dy = center_y - center_screen_y

        cmd_roll = int(np.clip(dx * Kp, -MAX_SPEED, MAX_SPEED))
        cmd_pitch = int(np.clip(-dy * Kp, -MAX_SPEED, MAX_SPEED))

        cv.circle(frame_resized, (center_x, center_y), 4, (0, 0, 255), -1)

    # Performans HUD'ı
    fps_color = (0, 255, 0) if fps >= 20 else (0, 0, 255)
    cv.putText(frame_resized, f"FPS: {fps:.1f} | LATENCY: {latency_ms:.1f}ms", 
               (15, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
    cv.putText(frame_resized, f"STATUS: {state} | ROLL: {cmd_roll:+03d} PITCH: {cmd_pitch:+03d}", 
               (15, 55), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # =========================================================================
    # 3. BÖLÜM: BURAK'IN KODU (Telemetri Loglama)
    # =========================================================================
    if current_time - last_log_time >= LOG_INTERVAL:
        last_log_time = current_time
        timestamp = time.strftime("%H:%M:%S")
        with open(log_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, state, center_x, center_y, cmd_roll, cmd_pitch, f"{fps:.1f}"])

    # Ekran merkezi nişangahı
    cv.drawMarker(frame_resized, (center_screen_x, center_screen_y), 
                  (255, 255, 255), cv.MARKER_CROSS, 14, 1)

    cv.imshow("IHA 4. Hafta - YOLO & Telemetri Pipeline", frame_resized)

    if cv.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap.release()
cv.destroyAllWindows()