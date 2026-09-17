import cv2 as cv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

frame_w, frame_h = 640,480
center_x = frame_w // 2
center_y = frame_h // 2
cap = cv.VideoCapture(0)
locked_id = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.resize(frame, (frame_w, frame_h))

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
            label = model.names[cls_id]

            if label != "person":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            area = w * h
            conf = float(box.conf[0])
            
            if locked_id is not None and locked_id == int(box.id[0]):
                target_w = w
                target_h = h
                target_x = x1
                target_y = y1
                target_conf = conf
                break

            elif locked_id == None:
                if area > max_area:
                    max_area = area
                    new_id = int(box.id[0])
                    new_x, new_y, new_w, new_h = x1, y1, w, h
                    new_conf = conf
                    
        if locked_id is not None:
            if target_x == None:
                locked_id = None
        else:
            if new_id is not None:
                locked_id = new_id
                target_x, target_y, target_h, target_w = new_x, new_y, new_h, new_w
                target_conf = new_conf   


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
        

    if target_x is not None:
        center_x = target_x + target_w // 2
        center_y = target_y + target_h // 2
        cv.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    cv.imshow("yolo", frame)

    if cv.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap.release()
cv.destroyAllWindows()