import cv2
import numpy as np
import os


# ELA Fun
def detect_tampering_ela(input_path, quality=90):
    original = cv2.imread(input_path)

    if original is None:
        raise ValueError("Image not found")

    temp_file = "temp_resaved.jpg"
    cv2.imwrite(temp_file, original, [cv2.IMWRITE_JPEG_QUALITY, quality])
    resaved = cv2.imread(temp_file)

    ela = cv2.absdiff(original, resaved)

    max_diff = np.max(ela)
    if max_diff == 0:
        max_diff = 1

    scale = 255.0 / max_diff
    ela = cv2.multiply(ela, scale)

    if os.path.exists(temp_file):
        os.remove(temp_file)

    return ela


# Dashboard 
def create_dashboard(original, ela, thresh, output):
    h, w = original.shape[:2]
    size = (400, 400)

    # Resize all images
    original_r = cv2.resize(original, size)
    ela_r = cv2.resize(ela, size)
    thresh_r = cv2.resize(thresh, size)
    output_r = cv2.resize(output, size)

    # Ensure same channels
    thresh_r = cv2.cvtColor(thresh_r, cv2.COLOR_GRAY2BGR)

    # Rows
    row1 = cv2.hconcat([original_r, ela_r])
    row2 = cv2.hconcat([thresh_r, output_r])

    dashboard = cv2.vconcat([row1, row2])

    dh, dw = dashboard.shape[:2]

    # Labels
    cv2.putText(dashboard, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(dashboard, "ELA Heatmap", (dw//2 + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(dashboard, "Mask", (10, dh//2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(dashboard, "Detected Regions", (dw//2 + 10, dh//2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return dashboard


# Main 

def analyze_image(path):
    original = cv2.imread(path)

    if original is None:
        print("Error: Image not found")
        return

    # ELA 
    ela = detect_tampering_ela(path, quality=90)

    # Grayscale
    gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)

    # Heatmap 
    ela_heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # Threshold
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Noise Removal
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Contours 
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = original.copy()

    for c in contours:
        if cv2.contourArea(c) > 100:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

    suspicious_pixels = np.sum(thresh == 255)
    total_pixels = thresh.size
    ratio = suspicious_pixels / total_pixels

    if ratio > 0.05:
        decision = "Possibly Tampered"
    else:
        decision = "Likely Authentic"

    print("=================================")
    print(f"Suspicious Ratio: {ratio:.4f}")
    print(f"Decision: {decision}")
    print("=================================")

    dashboard = create_dashboard(original, ela_heatmap, thresh, output)

    cv2.imshow("Image Tampering Detection System", dashboard)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# RUN
if __name__ == "__main__":
    analyze_image("test.jpg")