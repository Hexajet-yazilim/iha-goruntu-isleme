import cv2 as cv
import time
from ultralytics import YOLO
import csv                                      
from datetime import datetime

model = YOLO("yolov8n.pt")

cap = cv.VideoCapture(0)

log_file = open("flight_log.csv", "w", newline="", encoding="utf-8")       
csv_writer = csv.writer(log_file)                                         

csv_writer.writerow(["Zaman", "Durum", "Hedef_X", "Hedef_Y",             
                     "Roll_Komutu", "Pitch_Komutu"])                      

last_log_time = datetime.now()

while cap.isOpened():
    
    ret, frame = cap.read()
    if not ret:
        break
    ilk_t = time.time()
    frame = cv.resize(frame, (640, 480))

    results = model(frame, stream=True, conf=0.7) 

    target_x, target_y, target_w, target_h = None, None, None, None

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                target_w = x2 - x1
                target_h = y2 - y1
                target_x = x1
                target_y = y1

                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(frame, f"{label} {float(box.conf[0]):.2f}", 
                           (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                break

    if target_x is not None:
        current_time = datetime.now()                              
        elapsed_time = (current_time - last_log_time).total_seconds()  

        center_x = target_x + target_w // 2
        center_y = target_y + target_h // 2

        if elapsed_time >= 0.5:                                    
                    csv_writer.writerow([                                  
                        current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], 
                        #state,                                               
                        center_x,                                            
                        center_y,                                            
                        #round(komut_x, 2),                                   
                        #round(komut_y, 2)                                    
                    ])                                                       
        
                    log_file.flush()                                         
                    last_log_time = current_time

        cv.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    son_t = time.time()
    gecikme_sn = son_t - ilk_t
    fps = 1 / gecikme_sn if gecikme_sn > 0 else 0
    ms_latency = gecikme_sn * 1000

    fps_color = (0, 255, 0) if fps >= 30 else (0, 0, 255)
    
    cv.putText(frame, f"FPS: {fps:.1f} | LATENCY: {ms_latency:.0f}ms",
                (25, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)

    cv.imshow("hafta 4", frame)

    if cv.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

log_file.close()
cap.release()
cv.destroyAllWindows()