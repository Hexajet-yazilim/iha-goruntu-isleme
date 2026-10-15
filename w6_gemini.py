import cv2 as cv
from collections import deque
import numpy as np
import time
import csv
from datetime import datetime
from ultralytics import YOLO
import math   

model = YOLO("yolov8n.pt")
model.export(format="onnx") #başlarken koycaz
cap = cv.VideoCapture(0)

frame_w, frame_h = 640, 480
center_x = frame_w // 2
center_y = frame_h // 2

log_file = open("flight_log.csv", "w", newline="", encoding="utf-8")
csv_writer = csv.writer(log_file)
csv_writer.writerow(["Zaman", "Durum", "Hedef_X", "Hedef_Y", "Roll_Komutu", "Pitch_Komutu", "Failsafe"])
last_log_time = datetime.now()

state = "araniyor"
count = 0
MAX_FRAME = 15  # Burak: 15 kare kayıp toleransı
Kp = 0.05

# --- TAHA: PID GÜVENLİK SINIRLARI & FAILSAFE EŞİKLERİ ---
MAX_TILT_ANGLE = 25.0       # Derece cinsinden maksimum güvenli yatış açısı (Clamping)
MAX_SAFE_ERROR = 300.0      # Piksel cinsinden kabul edilebilir maksimum sapma
failsafe_active = False     # Acil durum / Pilot kontrol bayrağı

search_start_time = datetime.now()              
search_direction = 1                            
search_roll = 3                                 
search_period = 2.0                             
radar_angle = 0      

locked_id = None

# Hareketli Ortalama Filtresi için son N komutu tutacak kuyruklar
N_FRAMES = 5
roll_history = deque(maxlen=N_FRAMES)
pitch_history = deque(maxlen=N_FRAMES)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Kamera akışı alınamıyor.")
        break

    ilk_t = time.time()
    frame = cv.resize(frame, (frame_w, frame_h))

    # --- İLKER: MODEL VE TAKİPÇİ OPTİMİZASYONLARI ---
    # 1. imgsz=320: İşlemci yükünü 4 kat hafifletir.
    # 2. tracker="bytetrack.yaml": Düşük CPU tüketen Kalman tabanlı hafif takipçi.
    # 3. classes=[0]: Sadece insan sınıfını filtreler (arka planda 79 sınıf elenir).
    results = model.track(
        frame, 
        persist=True, 
        stream=True, 
        conf=0.7, 
        verbose=False,
        imgsz=320,
        tracker="bytetrack.yaml",
        classes=[0]
    )

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

            # classes=[0] sayesinde gelen her kutu garanti person'dır
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            area = w * h
            conf = float(box.conf[0])
            
            # Kilitli hedefi ara
            if locked_id is not None and locked_id == int(box.id[0]):
                target_w = w
                target_h = h
                target_x = x1
                target_y = y1
                target_conf = conf
                break  # Kilitli hedef bulundu, döngüden çık

            # Kilitli hedef yoksa en büyük adayı bul
            elif locked_id is None:
                if area > max_area:
                    max_area = area
                    new_id = int(box.id[0])
                    new_x, new_y, new_w, new_h = x1, y1, w, h
                    new_conf = conf

        # --- BURAK: 15 KARE KAYIP HEDEF TOLERANSI MANTIĞI ---
        # Hedef bu karede görünmese bile locked_id ANINDA SIFIRLANMAZ.
        # Sadece 15 kare üst üste hedef gelmezse kilit düşürülür.
        if locked_id is None and new_id is not None:
            locked_id = new_id
            target_x, target_y, target_w, target_h = new_x, new_y, new_w, new_h
            target_conf = new_conf   

        # Çizim ve HUD
        if target_x is not None:
            cx = target_x + target_w // 2
            cy = target_y + target_h // 2

            cv.rectangle(frame, (target_x, target_y), (target_x + target_w, target_y + target_h), (0, 255, 0), 2)
            cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv.line(frame, (center_x, center_y), (cx, cy), (0, 255, 255), 1)
            cv.putText(frame, f"LOCKED ID:{locked_id} {target_conf:.2f}", 
                        (target_x, target_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        else:
            cv.putText(frame, "STATUS: SEARCHING", (15, 30), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv.drawMarker(frame, (center_x, center_y), (255, 255, 255), cv.MARKER_CROSS, 16, 1)

    detected = (target_x is not None)

    if detected:
        state = "bulundu"
        count = 0  # Burak: Hedef görüldüğü an kayıp sayacı sıfırlanır
        center_x = target_x + target_w // 2
        center_y = target_y + target_h // 2
    else:
        count += 1
        if count < MAX_FRAME:
            state = "araniyor"
            # Burak: 15 kare dolana kadar kilitli ID korunur (Kör takip / Tolerans modu)
        else:
            state = "yok"
            locked_id = None  # Burak: 15 kare boyunca bulunamadı, kilit düşürüldü
            if count == MAX_FRAME:
                search_start_time = datetime.now()
            # Hedef kayıpsa geçmişi temizle
            roll_history.clear()
            pitch_history.clear()
        center_x, center_y = None, None

    radar_x, radar_y, radar_size = 520, 20, 100
    overlay = frame.copy()
    cv.rectangle(overlay, (radar_x, radar_y), (radar_x + radar_size, radar_y + radar_size), (0, 0, 0), -1)
    frame = cv.addWeighted(frame, 0.7, overlay, 0.3, 0)

    radar_center_x = radar_x + radar_size // 2
    radar_center_y = radar_y + radar_size // 2

    cv.line(frame, (radar_center_x - 8, radar_center_y), (radar_center_x + 8, radar_center_y), (255, 255, 255), 2)
    cv.line(frame, (radar_center_x, radar_center_y - 8), (radar_center_x, radar_center_y + 8), (255, 255, 255), 2)

    if state == "bulundu":
        status_color = (0, 255, 0)
        status_text = "STATUS: TARGET LOCKED"
    elif state == "araniyor":
        status_color = (0, 255, 255)
        status_text = f"STATUS: SEARCHING ({MAX_FRAME - count})"
    else:
        status_color = (0, 0, 255)
        status_text = "STATUS: TARGET LOST"

        current_search_time = datetime.now()        
        elapsed_search_time = (                   
            current_search_time - search_start_time
        ).total_seconds()

        if elapsed_search_time >= search_period:   
            search_direction *= -1                 
            search_start_time = current_search_time 

        search_command = search_roll * search_direction  

        cv.putText(                                
            frame,
            f"SEARCH ROLL: {search_command}",
            (25, 80),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2
        )

        cv.putText(                                
            frame,
            "SEARCH PATTERN ACTIVE",
            (25, 110),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2
        )

        # radar çizgisi
        radar_angle += 4                           

        if radar_angle >= 360:                     
            radar_angle = 0                        

        angle_rad = math.radians(radar_angle)      
        sweep_length = radar_size // 2 - 5         

        sweep_x = int(radar_center_x + math.cos(angle_rad) * sweep_length)
        sweep_y = int(radar_center_y + math.sin(angle_rad) * sweep_length)

        cv.line(frame, (radar_center_x, radar_center_y), (sweep_x, sweep_y), (0, 255, 255), 2)

    cv.putText(frame, status_text, (25, 45), cv.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    komut_x, komut_y = 0.0, 0.0

    if detected:
        cv.rectangle(frame, (target_x, target_y), (target_x + target_w, target_y + target_h), (0, 255, 0), 2)
        cv.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

        dx = center_x - 320
        dy = center_y - 240

        # --- TAHA: FAILSAFE VE GÜVENLİK SINIRLAMASI (CLAMPING) ---
        # Anormal piksel sapması veya FPS çökmesi durumunda Failsafe tetiklenir
        if abs(dx) > MAX_SAFE_ERROR or abs(dy) > MAX_SAFE_ERROR:
            failsafe_active = True
            komut_x, komut_y = 0.0, 0.0  # Drone'u tehlikeye atmamak için komutları sıfırla (Brake/Hover)
            cv.putText(frame, "FAILSAFE: ERROR OUT OF BOUNDS", (25, 140),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            failsafe_active = False
            raw_x = dx * Kp
            raw_y = dy * Kp

            # Clamping: Açı komutunu sınırla (-25 ile +25 derece arası)
            komut_x = max(-MAX_TILT_ANGLE, min(MAX_TILT_ANGLE, raw_x))
            komut_y = max(-MAX_TILT_ANGLE, min(MAX_TILT_ANGLE, raw_y))

            # MOVING AVERAGE FİLTRESİ
            roll_history.append(komut_x)
            pitch_history.append(komut_y)

            smoothed_roll = sum(roll_history) / len(roll_history)
            smoothed_pitch = sum(pitch_history) / len(pitch_history)

            komut_x = smoothed_roll
            komut_y = smoothed_pitch

        radar_dx = int(dx * radar_size / 640)
        radar_dy = int(dy * radar_size / 480)
        radar_target_x = radar_center_x + radar_dx
        radar_target_y = radar_center_y + radar_dy
        cv.circle(frame, (radar_target_x, radar_target_y), 4, (0, 0, 255), -1)

        cv.putText(frame, f"YON (roll:{komut_x:.2f} pitch:{komut_y:.2f})",
                   (target_x, target_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    current_time = datetime.now()
    elapsed_time = (current_time - last_log_time).total_seconds()

    if elapsed_time >= 0.5:
        cx_log = center_x if center_x is not None else "-"
        cy_log = center_y if center_y is not None else "-"

        csv_writer.writerow([
            current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            state,
            cx_log,
            cy_log,
            round(komut_x, 2),
            round(komut_y, 2),
            failsafe_active
        ])
        log_file.flush()
        last_log_time = current_time

    son_t = time.time()
    gecikme_sn = son_t - ilk_t
    fps = 1 / gecikme_sn if gecikme_sn > 0 else 0
    ms_latency = gecikme_sn * 1000

    fps_color = (0, 255, 0) if fps >= 20 else (0, 0, 255)
    cv.putText(frame, f"FPS: {fps:.1f} | LATENCY: {ms_latency:.0f}ms",
               (25, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)

    cv.imshow("IHA Otonom Canli Takip - Birlestirilmis Ana Surum", frame)

    if cv.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

log_file.close()
cap.release()
cv.destroyAllWindows()