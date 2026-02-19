import cv2
import numpy as np

# Fixed HSV thresholds (from your tuned values)
LOWER_HSV = np.array([4, 40, 40], dtype=np.uint8)   # [LH, LS, LV]
UPPER_HSV = np.array([45, 255, 255], dtype=np.uint8) # [UH, US, UV]

# Other tuned parameters
HUE_TOL = 20
MIN_AREA = 5000
SOLIDITY_THRESH = 0.60
KERNEL_SIZE = 21  # should be odd

def ensure_odd(x):
    return x if x % 2 == 1 else x+1

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    k = ensure_odd(KERNEL_SIZE)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Apply mask
        mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        target_hue = (LOWER_HSV[0] + UPPER_HSV[0]) / 2.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
            solidity = area / hull_area
            if solidity < SOLIDITY_THRESH:
                continue

            mask_cnt = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
            mean_hsv = cv2.mean(hsv, mask=mask_cnt)
            mean_h = mean_hsv[0]

            dh = abs(mean_h - target_hue)
            dh = min(dh, 180 - dh)

            if dh <= HUE_TOL:
                candidates.append((cnt, area))

        selected_cnt = None
        if candidates:
            selected_cnt = max(candidates, key=lambda x: x[1])[0]

        out = frame.copy()
        if selected_cnt is not None:
            hull = cv2.convexHull(selected_cnt)
            cv2.drawContours(out, [hull], -1, (0, 255, 0), 3)
            x, y, w, h = cv2.boundingRect(selected_cnt)
            cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(out, "PEAR", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        else:
            cv2.putText(out, "No pear detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.imshow("Mask", mask)
        cv2.imshow("Pear Detection", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
