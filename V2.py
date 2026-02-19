import cv2
import numpy as np

cap = cv2.VideoCapture(0)

kernel = np.ones((5, 5), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -----------------------------
    # STEP 1: PEAR SEGMENTATION
    # -----------------------------
    # Pear is brighter than background
    _, thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        cv2.imshow("Pear Inspection", frame)
        continue

    # Take largest object = pear
    pear_cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(pear_cnt)

    if area < 5000:
        continue

    x, y, w, h = cv2.boundingRect(pear_cnt)
    pear_roi = frame[y:y+h, x:x+w]
    pear_hsv = hsv[y:y+h, x:x+w]

    # -----------------------------
    # STEP 2: ROT / MOLD DETECTION
    # -----------------------------
    # Rotten = low V + low S (dark + dull)
    rot_lower = np.array([0, 0, 0])
    rot_upper = np.array([180, 90, 90])

    rot_mask = cv2.inRange(pear_hsv, rot_lower, rot_upper)
    rot_mask = cv2.morphologyEx(rot_mask, cv2.MORPH_OPEN, kernel)

    rot_pixels = cv2.countNonZero(rot_mask)
    pear_pixels = w * h

    rot_ratio = rot_pixels / pear_pixels

    # -----------------------------
    # DECISION
    # -----------------------------
    if rot_ratio > 0.05:
        label = "BAD"
        color = (0, 0, 255)
    else:
        label = "GOOD"
        color = (0, 255, 0)

    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frame, label, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Pear Inspection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
