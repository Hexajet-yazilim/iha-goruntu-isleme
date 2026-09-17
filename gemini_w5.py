import csv
import math
import time
from collections import deque
import cv2 as cv
from ultralytics import YOLO

# =============================================================================
# 1. MODÜL: PID VE HAREKETLİ ORTALAMA FİLTRESİ (TAHA)
# =============================================================================
class FlightController:
    def __init__(self, kp=0.25, ki=0.01, kd=0.08, window_size=5):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.prev_error_x = 0
        self.prev_error_y = 0
        self.integral_x = 0
        self.integral_y = 0

        # Titreşimi önleyen hareketli ortalama kuyrukları
        self.roll_queue = deque(maxlen=window_size)
        self.pitch_queue = deque(maxlen=window_size)

    def update(self, error_x, error_y, dt):
        if dt <= 0:
            dt = 0.033

        # PID Hesaplamaları
        self.integral_x += error_x * dt
        self.integral_y += error_y * dt
        derivative_x = (error_x - self.prev_error_x) / dt
        derivative_y = (error_y - self.prev_error_y) / dt

        raw_roll = (self.kp * error_x) + (self.ki * self.integral_x) + (self.kd * derivative_x)
        raw_pitch = (self.kp * error_y) + (self.ki * self.integral_y) + (self.kd * derivative_y)

        self.prev_error_x = error_x
        self.prev_error_y = error_y

        # Moving Average Filtresi
        self.roll_queue.append(raw_roll)
        self.pitch_queue.append(raw_pitch)

        filtered_roll = sum(self.roll_queue) / len(self.roll_queue)
        filtered_pitch = sum(self.pitch_queue) / len(self.pitch_queue)

        return filtered_roll, filtered_pitch

    def reset_integral(self):
        self.integral_x = 0
        self.integral_y = 0
        self.roll_queue.clear()
        self.pitch_queue.clear()


# =============================================================================
# 2. MODÜL: ARAMA PATERNI, HUD VE TELEMETRİ LOGLAMA (BURAK)
# =============================================================================
class FlightLoggerAndHUD:
    def __init__(self, log_path="flight_log.csv"):
        self.log_file = open(log_path, mode="w", newline="")
        self.writer = csv.writer(self.log_file)
        self.writer.writerow(["timestamp", "state", "target_id", "error_x", "error_y", "cmd_roll", "cmd_pitch", "fps"])
        self.search_angle = 0.0

    def get_search_commands(self):
        """Hedef kaybolduğunda periyodik dairesel/yaw arama paterni üretir."""
        self.search_angle += 0.05
        search_yaw = math.sin(self.search_angle) * 15.0
        return search_yaw

    def draw_search_pattern(self, frame, center_x, center_y):
        sweep_radius = 50
        line_x = int(center_x + sweep_radius * math.cos(self.search_angle))
        line_y = int(center_y + sweep_radius * math.sin(self.search_angle))
        
        cv.circle(frame, (center_x, center_y), sweep_radius, (0, 0, 255), 1)
        cv.line(frame, (center_x, center_y), (line_x, line_y), (0, 0, 255), 2)
        cv.putText(frame, "STATUS: SEARCH PATTERN ACTIVE", (15, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def log(self, state, target_id, error_x, error_y, cmd_roll, cmd_pitch, fps):
        self.writer.writerow([
            f"{time.time():.3f}", state, target_id if target_id is not None else -1,
            f"{error_x:.1f}", f"{error_y:.1f}", f"{cmd_roll:.2f}", f"{cmd_pitch:.2f}", f"{fps:.1f}"
        ])

    def close(self):
        self.log_file.close()


# =============================================================================
# 3. ANA ÇALIŞTIRMA BORU HATTI (İLKER / LEAD ENTEGRASYON)
# =============================================================================
def main():
    model = YOLO("yolov8n.pt")
    controller = FlightController()
    hud_logger = FlightLoggerAndHUD()

    frame_w, frame_h = 640, 480
    center_x = frame_w // 2
    center_y = frame_h // 2

    cap = cv.VideoCapture(0)
    locked_id = None

    prev_time = time.time()

    while cap.isOpened():
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time
        fps = 1.0 / dt if dt > 0 else 0.0

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.resize(frame, (frame_w, frame_h))

        # İlker: YOLOv8 Takip ve Kilit
        results = model.track(frame, persist=True, stream=True, conf=0.6, verbose=False)

        target_x, target_y, target_w, target_h = None, None, None, None
        target_conf = 0.0

        new_id = None
        new_x, new_y, new_w, new_h = None, None, None, None
        new_conf = 0.0
        max_area = 0

        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue

            for box in r.boxes:
                if box.id is None:
                    continue

                cls_id = int(box.cls[0])
                if model.names[cls_id] != "person":
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                area = w * h
                conf = float(box.conf[0])

                if locked_id is not None and locked_id == int(box.id[0]):
                    target_x, target_y, target_w, target_h = x1, y1, w, h
                    target_conf = conf
                    break

                elif locked_id is None:
                    if area > max_area:
                        max_area = area
                        new_id = int(box.id[0])
                        new_x, new_y, new_w, new_h = x1, y1, w, h
                        new_conf = conf

        # Durum ve Kilit Mantığı
        if locked_id is not None:
            if target_x is None:
                locked_id = None
                controller.reset_integral()
        else:
            if new_id is not None:
                locked_id = new_id
                target_x, target_y, target_w, target_h = new_x, new_y, new_w, new_h
                target_conf = new_conf

        # Karar ve Kontrol Komutları
        if target_x is not None:
            state = "LOCKED"
            cx = target_x + target_w // 2
            cy = target_y + target_h // 2

            error_x = cx - center_x
            error_y = cy - center_y

            cmd_roll, cmd_pitch = controller.update(error_x, error_y, dt)

            # Görselleştirme
            cv.rectangle(frame, (target_x, target_y), (target_x + target_w, target_y + target_h), (0, 255, 0), 2)
            cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv.line(frame, (center_x, center_y), (cx, cy), (0, 255, 255), 1)
            cv.putText(frame, f"LOCKED ID:{locked_id} {target_conf:.2f}",
                       (target_x, target_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        else:
            state = "SEARCHING"
            error_x, error_y = 0.0, 0.0
            cmd_roll = hud_logger.get_search_commands()
            cmd_pitch = 0.0

            hud_logger.draw_search_pattern(frame, center_x, center_y)

        # Telemetri HUD & Loglama
        cv.drawMarker(frame, (center_x, center_y), (255, 255, 255), cv.MARKER_CROSS, 16, 1)
        cv.putText(frame, f"FPS: {fps:.1f}", (frame_w - 110, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.putText(frame, f"CMD: R:{cmd_roll:.1f} P:{cmd_pitch:.1f}", (15, frame_h - 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        hud_logger.log(state, locked_id, error_x, error_y, cmd_roll, cmd_pitch, fps)

        cv.imshow("UAV Central Tracking Pipeline", frame)
        if cv.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

    hud_logger.close()
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()