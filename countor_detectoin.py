import cv2
import numpy as np
import tensorflow as tf

# === PATH TO YOUR MODEL ===
MODEL_PATH = r"D:\Engineering\Graduation_project\Vision\Google Teachable Machoine\vww_96_grayscale_quantized.tflite"

# === Load TFLite model ===
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Utility functions ===
def get_largest_contour_from_mask(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 500:
        return None
    return c

def process_frame(frame):
    orig = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # HSV range for yellow-green pears (adjust to your lighting)
    lower_yellow = np.array([10, 40, 40])
    upper_yellow = np.array([45, 255, 255])
    mask_color = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21))
    mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel)
    mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_OPEN, kernel)

    cnt = get_largest_contour_from_mask(mask_color)
    mask_used = mask_color

    if cnt is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7,7), 0)
        _, mask_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ch, cw = mask_thresh.shape
        if np.mean(gray[ch//4:3*ch//4, cw//4:3*cw//4]) < np.mean(gray):
            mask_thresh = cv2.bitwise_not(mask_thresh)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
        mask_thresh = cv2.morphologyEx(mask_thresh, cv2.MORPH_CLOSE, kernel2)
        mask_thresh = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel2)
        cnt = get_largest_contour_from_mask(mask_thresh)
        mask_used = mask_thresh

    if cnt is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        cnt = get_largest_contour_from_mask(edges)
        mask_used = edges

    output = orig.copy()
    if cnt is not None:
        hull = cv2.convexHull(cnt)
        cv2.drawContours(output, [hull], -1, (0,255,0), 3)  # Green contour for pear

    return output, mask_used, cnt

# === Main Loop ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Step 1: AI Classification ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (96, 96))
    img = np.expand_dims(img, axis=(0, -1)).astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    pear_prob, not_pear_prob = prediction
    label = "Pear 🍐" if pear_prob > not_pear_prob else "Not Pear 🚫"

    # --- Step 2: Contour-based Localization ---
    out_frame, mask, cnt = process_frame(frame)

    # --- Step 3: Infection detection (inside the pear contour only) ---
    if label == "Pear 🍐" and cnt is not None:
        x, y, w, h = cv2.boundingRect(cnt)
        pear_roi = frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(pear_roi, cv2.COLOR_BGR2HSV)

        # Detect darker / brownish spots (infection)
        lower_infect = np.array([0, 40, 0])      # dark red/brown start
        upper_infect = np.array([25, 255, 120])  # brown/yellowish low brightness
        mask_infect = cv2.inRange(hsv_roi, lower_infect, upper_infect)

        kernel_inf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
        mask_infect = cv2.morphologyEx(mask_infect, cv2.MORPH_CLOSE, kernel_inf)
        mask_infect = cv2.morphologyEx(mask_infect, cv2.MORPH_OPEN, kernel_inf)

        # Find infection contours and draw red outlines
        contours, _ = cv2.findContours(mask_infect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > 150:
                cx, cy, cw, ch = cv2.boundingRect(c)
                cv2.rectangle(out_frame, (x+cx, y+cy), (x+cx+cw, y+cy+ch), (0,0,255), 2)

    # --- Step 4: Labels and display ---
    color = (0, 255, 0) if label == "Pear 🍐" else (0, 0, 255)
    cv2.putText(out_frame, f"{label} ({pear_prob:.2f})", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Pear Detector", out_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
