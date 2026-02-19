import cv2
import numpy as np
import math

def nothing(x):
    pass

# Window & trackbars for tuning
cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
cv2.createTrackbar("LH", "Trackbars", 10, 180, nothing)
cv2.createTrackbar("LS", "Trackbars", 40, 255, nothing)
cv2.createTrackbar("LV", "Trackbars", 40, 255, nothing)
cv2.createTrackbar("UH", "Trackbars", 45, 180, nothing)
cv2.createTrackbar("US", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("UV", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("HueTol", "Trackbars", 20, 90, nothing)    # tolerance around mean hue
cv2.createTrackbar("MinArea", "Trackbars", 5000, 200000, nothing)
cv2.createTrackbar("Solidity%", "Trackbars", 60, 100, nothing) # solidity threshold (percent)
cv2.createTrackbar("Kernel", "Trackbars", 21, 51, nothing)     # morphological kernel (odd)

def ensure_odd(x):
    return x if x % 2 == 1 else x+1

def get_largest_contour(cnts, min_area):
    if not cnts:
        return None
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    return max(cnts, key=cv2.contourArea) if cnts else None

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # optional resize for speed (uncomment if needed)
    # frame = cv2.resize(frame, (960, int(frame.shape[0]*960/frame.shape[1])))

    lh = cv2.getTrackbarPos("LH", "Trackbars")
    ls = cv2.getTrackbarPos("LS", "Trackbars")
    lv = cv2.getTrackbarPos("LV", "Trackbars")
    uh = cv2.getTrackbarPos("UH", "Trackbars")
    us = cv2.getTrackbarPos("US", "Trackbars")
    uv = cv2.getTrackbarPos("UV", "Trackbars")
    hue_tol = cv2.getTrackbarPos("HueTol", "Trackbars")
    min_area = cv2.getTrackbarPos("MinArea", "Trackbars")
    solidity_thresh = cv2.getTrackbarPos("Solidity%", "Trackbars") / 100.0
    k = cv2.getTrackbarPos("Kernel", "Trackbars")
    k = ensure_odd(max(3, k))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([lh, ls, lv], dtype=np.uint8)
    upper = np.array([uh, us, uv], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    # Morphological cleanup: open (remove small specks) then close (fill small holes)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours on mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    target_hue = (lh + uh) / 2.0  # center hue of trackbar range

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < max(1, min_area):
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue
        solidity = area / hull_area

        # Reject shapes that are too concave or too small
        if solidity < solidity_thresh:
            continue

        # Compute mean HSV inside this contour
        mask_cnt = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
        mean_hsv = cv2.mean(hsv, mask=mask_cnt)
        mean_h = mean_hsv[0]  # hue in [0..180]

        # hue distance on circular hue space
        dh = abs(mean_h - target_hue)
        dh = min(dh, 180 - dh)

        if dh <= hue_tol:
            candidates.append((cnt, area, solidity, mean_h))

    selected_cnt = None
    # Prefer the largest color-matching candidate
    if candidates:
        selected_cnt = max(candidates, key=lambda x: x[1])[0]
    else:
        # fallback: use the largest contour by area (if it passes basic area test)
        all_big = [c for c in contours if cv2.contourArea(c) >= max(1, min_area)]
        if all_big:
            selected_cnt = max(all_big, key=cv2.contourArea)

    out = frame.copy()
    if selected_cnt is not None:
        hull = cv2.convexHull(selected_cnt)
        cv2.drawContours(out, [hull], -1, (0, 255, 0), 3)  # green outline
        x,y,w,h = cv2.boundingRect(selected_cnt)
        cv2.rectangle(out, (x,y), (x+w, y+h), (0,255,0), 1)
        cv2.putText(out, "PEAR", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    else:
        cv2.putText(out, "No pear detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # Optional: show mask and the result
    cv2.imshow("Mask (tuned)", mask)
    cv2.imshow("Pear Detection", out)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        # save sample
        cv2.imwrite("pear_detect_snapshot.png", out)
        print("Saved pear_detect_snapshot.png")

cap.release()
cv2.destroyAllWindows()
