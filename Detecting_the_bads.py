print("--- STEP 1: Starting Python Script ---")

import cv2
import numpy as np
import os

# --- IMPORT TENSORFLOW ---
try:
    import tflite_runtime.interpreter as tflite
    print("SUCCESS: Using lightweight tflite_runtime")
except ImportError:
    print("WARNING: tflite_runtime not found. Falling back to full TensorFlow.")
    import tensorflow.lite as tflite
    print("SUCCESS: Full TensorFlow loaded.")

def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip().split(' ', 1)[-1] for line in f.readlines()]

def check_for_rot_blobs(roi_image):
    """
    New Logic: Looks for SOLID BLOBS of rot, not just random pixels.
    Returns: (True/False, Max_Spot_Size)
    """
    hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    height, width = roi_image.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    # 1. Aggressive Rot Colors (Dark Black + Brown)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80]) # Increased Value to 80 to catch lighter rot
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    lower_brown = np.array([0, 60, 20])   # Increased Saturation to 60 to avoid shadows
    upper_brown = np.array([25, 255, 200])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    rot_mask = cv2.bitwise_or(mask_black, mask_brown)

    # 2. Clean up noise (Close small holes)
    kernel = np.ones((5, 5), np.uint8)
    rot_mask = cv2.morphologyEx(rot_mask, cv2.MORPH_CLOSE, kernel)
    
    # 3. Find Contours of the ROT SPOTS
    contours, _ = cv2.findContours(rot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_blob_area = 0
    is_rotten = False
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # FILTER 1: Size
        # A rot spot must be bigger than 100 pixels to matter
        if area > 100:
            
            # FILTER 2: Location (The "Stem Filter")
            # We calculate the center of this specific spot
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Calculate distance from the CENTER of the pear
                distance_from_center = np.sqrt((cX - center_x)**2 + (cY - center_y)**2)
                
                # If the spot is near the edge (stem area), ignore it.
                # Allowed distance is 45% of the pear's width
                allowed_radius = width * 0.45 
                
                if distance_from_center < allowed_radius:
                    # It is big AND it is in the center/body
                    max_blob_area = max(max_blob_area, area)
                    is_rotten = True
                    
                    # Draw this spot on the ROI for the Debug Window
                    cv2.drawContours(roi_image, [cnt], -1, (0, 0, 255), 2)

    return is_rotten, max_blob_area, roi_image

def classify_pear(interpreter, input_details, output_details, roi_image):
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    height, width = input_shape[1], input_shape[2]
    channels = input_shape[3] if len(input_shape) > 3 else 1
    
    resized_roi = cv2.resize(roi_image, (width, height))
    
    if channels == 1:
        if len(resized_roi.shape) == 3:
            input_data = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2GRAY)
            input_data = np.expand_dims(input_data, axis=-1)
    else:
        input_data = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2RGB)

    if input_dtype == np.uint8:
        input_data = np.array(input_data, dtype=np.uint8)
    elif input_dtype == np.float32:
        input_data = (np.float32(input_data) / 127.5) - 1.0
    else:
        input_data = np.array(input_data, dtype=input_dtype)

    input_data = np.expand_dims(input_data, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

def main():
    base_folder = r"D:\Engineering\Graduation_project\Vision\Jeston Nano"
    MODEL_PATH = os.path.join(base_folder, "vww_96_grayscale_quantized.tflite")
    LABELS_PATH = os.path.join(base_folder, "labels.txt")
    
    if not os.path.exists(MODEL_PATH):
        print("ERROR: Paths are wrong. Please check folder.")
        return

    labels = load_labels(LABELS_PATH)
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. PRE-PROCESSING
        blurred = cv2.GaussianBlur(frame, (13, 13), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 2. LOCALIZE PEAR (Find the fruit)
        # Green Skin
        lower_green = np.array([20, 40, 40])
        upper_green = np.array([100, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Brown Skin (for detection)
        lower_brown_detect = np.array([0, 20, 10])
        upper_brown_detect = np.array([30, 255, 200])
        mask_brown_detect = cv2.inRange(hsv, lower_brown_detect, upper_brown_detect)

        combined_mask = cv2.bitwise_or(mask_green, mask_brown_detect)
        kernel = np.ones((15, 15), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pear_count = 0
        debug_collage = None # To store images of detected pears
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2000:
                hull = cv2.convexHull(cnt)
                x, y, w, h = cv2.boundingRect(hull)
                
                if w > 10 and h > 10:
                    pear_roi = frame[y:y+h, x:x+w].copy() # Work on a copy
                    
                    # --- NEW LOGIC: BLOB CHECK ---
                    # Checks for large connected spots in the center
                    is_rotten, spot_size, debug_roi = check_for_rot_blobs(pear_roi)
                    
                    final_status = "UNKNOWN"
                    color = (0, 255, 0)
                    
                    # PRIORITY 1: The "Big Spot" Rule
                    if is_rotten:
                        final_status = "BAD (Spot)"
                        color = (0, 0, 255) # Red
                    
                    # PRIORITY 2: Ask AI (Only if no obvious spots found)
                    else:
                        predictions = classify_pear(interpreter, input_details, output_details, pear_roi)
                        max_score_index = np.argmax(predictions)
                        raw_label = labels[max_score_index].lower()
                        
                        if "bad" in raw_label or "rot" in raw_label:
                            final_status = "BAD (AI)"
                            color = (0, 0, 255)
                        else:
                            final_status = "GOOD"
                            color = (0, 255, 0)

                    # Draw Main Box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    cv2.rectangle(frame, (x, y-45), (x+160, y), color, -1)
                    cv2.putText(frame, final_status, (x + 5, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    pear_count += 1
                    
                    # --- BUILD DEBUG WINDOW ---
                    # We stack the analyzed pear images to show you what the computer sees
                    if debug_collage is None:
                        debug_collage = cv2.resize(debug_roi, (200, 200))
                    else:
                        # Append next pear horizontally
                        resized_next = cv2.resize(debug_roi, (200, 200))
                        # Limit collage width to prevent crash if too many pears
                        if debug_collage.shape[1] < 1000: 
                            debug_collage = np.hstack((debug_collage, resized_next))

        cv2.putText(frame, f"Total Pears: {pear_count}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow("Smart Pear Sorter", frame)
        
        # Show the "Brain" View - RED OUTLINES mean rot was detected there
        if debug_collage is not None:
            cv2.imshow("Debug View (Red=Rot)", debug_collage)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()