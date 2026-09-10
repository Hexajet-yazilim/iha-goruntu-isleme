import cv2 as cv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

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
        center_x = target_x + target_w // 2
        center_y = target_y + target_h // 2
        cv.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    cv.imshow("yolo", frame)

    if cv.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap.release()
cv.destroyAllWindows()