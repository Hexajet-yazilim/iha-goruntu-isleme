import cv2 as cv
import numpy as np
import csv                                      # 
from datetime import datetime                  # 

cap = cv.VideoCapture(0)
kernel = np.ones((5, 5), np.uint8)

state = "araniyor"           
count = 0             
MAX_FRAME = 15   
detected = False

Kp = 0.05

log_file = open("flight_log.csv", "w", newline="", encoding="utf-8")       # 
csv_writer = csv.writer(log_file)                                         # 

csv_writer.writerow(["Zaman", "Durum", "Hedef_X", "Hedef_Y",             # 
                     "Roll_Komutu", "Pitch_Komutu"])                     # 

last_log_time = datetime.now()                                            # 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Kamera akışı alınamıyor.")
        break
    
    frame_resized = cv.resize(frame, (640, 480))
    hsv = cv.cvtColor(frame_resized, cv.COLOR_BGR2HSV)
    lower_color = np.array([90, 20, 20])
    upper_color = np.array([140, 255, 255])
    mask = cv.inRange(hsv, lower_color, upper_color)

    mask = cv.erode(mask, kernel, iterations=1)
    mask = cv.dilate(mask, kernel, iterations=1)

    radar_x = 520
    radar_y = 20
    radar_size = 100

    overlay = frame_resized.copy()

    cv.rectangle(
        overlay,
        (radar_x, radar_y),
        (radar_x + radar_size, radar_y + radar_size),
        (0, 0, 0),
        -1
    )
    
    frame_resized = cv.addWeighted(frame_resized, 0.7, overlay, 0.3, 0)
  
    radar_center_x = radar_x + radar_size // 2
    radar_center_y = radar_y + radar_size // 2

    cv.line(
        frame_resized,
        (radar_center_x - 8, radar_center_y),
        (radar_center_x + 8, radar_center_y),
        (255, 255, 255),
        2
    )
    cv.line(
        frame_resized,
        (radar_center_x, radar_center_y - 8),
        (radar_center_x, radar_center_y + 8),
        (255, 255, 255),
        2
    )

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    target_x, target_y, target_w, target_h = None, None, None, None

    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        detected = True
        if cv.contourArea(largest_contour) > 500:
            
            target_x, target_y, target_w, target_h = cv.boundingRect(largest_contour)
            
            center_x = target_x + target_w // 2
            center_y = target_y + target_h // 2
    else:
        detected = False

    if detected == True:
        state = "bulundu"
        count = 0
    else:
        count += 1
        if count < MAX_FRAME:
            state = "araniyor"
        else:
            state = "yok"

    if state == "bulundu":
        status_color = (0, 255, 0)    
        status_text = "STATUS: TARGET LOCKED"
    elif state == "araniyor":
        status_color = (0, 255, 255)  
        status_text = f"STATUS: SEARCHING ({MAX_FRAME - count})"
    else:
        status_color = (0, 0, 255)    
        status_text = "STATUS: TARGET LOST"

    cv.putText(frame_resized, status_text, (25, 45), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)    

    if target_x is not None and target_y is not None:
        cv.rectangle(frame_resized, (target_x, target_y), 
                     (target_x + target_w, target_y + target_h), (0, 0, 255), 2)

        cv.circle(frame_resized, (center_x, center_y), 5, (0, 255, 0), -1)

        dx = center_x - 320
        dy = center_y - 240

        komut_x = dx * Kp
        komut_y = dy * Kp

        komut_x = max(-8, min(8, komut_x))
        komut_y = max(-8, min(8, komut_y))

        cv.rectangle(
            frame_resized,
            (target_x, target_y),
            (target_x + target_w, target_y + target_h),
            (0, 255, 0),
            2
        )

        cv.circle(
            frame_resized,
            (center_x, center_y),
            5,
            (0, 255, 0),
            -1
        )  

        radar_dx = int(dx * radar_size / 640)
        radar_dy = int(dy * radar_size / 480)

        radar_target_x = radar_center_x + radar_dx
        radar_target_y = radar_center_y + radar_dy

        
        cv.circle(
            frame_resized,
            (radar_target_x, radar_target_y),
            4,
            (0, 0, 255),
            -1
        )

        if state == "bulundu":
            cv.putText(frame_resized, f"yön (roll:{komut_x:.2f} pitch:{komut_y:.2f})",
                                           (target_x, target_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # csv

            current_time = datetime.now()                              # 
            elapsed_time = (current_time - last_log_time).total_seconds()  #

            if elapsed_time >= 0.5:                                    # 
                csv_writer.writerow([                                  # 
                    current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], # 
                    state,                                               # 
                    center_x,                                            # 
                    center_y,                                            # 
                    round(komut_x, 2),                                   # 
                    round(komut_y, 2)                                    # 
                ])                                                       # 

                log_file.flush()                                         # 
                last_log_time = current_time                             # 

    cv.imshow("IHA Otonom Canli Takip - 2. Hafta", frame_resized)
    
    cv.imshow("Ilker - Maske Girdisi", mask)

    k = cv.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

log_file.close()                                                         # 