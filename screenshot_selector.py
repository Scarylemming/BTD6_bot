import cv2
import numpy as np
import os
import glob

# Initialize variables
drawing = False
ix, iy = -1, -1
fx, fy = -1, -1

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            fx, fy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        print(f"Region Coordinates: (x={min(ix, fx)}, y={min(iy, fy)}, width={abs(fx-ix)}, height={abs(fy-iy)})")
        cv2.destroyAllWindows()

def get_latest_screenshot(folder_path):
    # List all PNG files in the folder
    screenshot_files = glob.glob(os.path.join(folder_path, "*.png"))
    
    # Sort by modification time (newest first)
    screenshot_files.sort(key=os.path.getmtime, reverse=True)
    
    if not screenshot_files:
        raise FileNotFoundError("No screenshots found in the folder!")
    
    return screenshot_files[0]  # Return the newest screenshot

# Example usage (using raw string for Windows paths)
screenshot_folder = r"images/screenshots"  # <- FIXED (backslashes + raw string)
latest_screenshot_path = get_latest_screenshot(screenshot_folder)

# Load the image properly (using OpenCV)
screenshot = cv2.imread(latest_screenshot_path)

if screenshot is None:
    raise FileNotFoundError("Failed to load the screenshot!")

# Get image dimensions
height, width = screenshot.shape[:2]
print(f"Image dimensions: {width}x{height}")

# Create a window and bind the mouse callback
cv2.namedWindow("Select Region", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Select Region", draw_rectangle)

while True:
    img = screenshot.copy()
    if ix != -1 and iy != -1 and fx != -1 and fy != -1:
        cv2.rectangle(img, (ix, iy), (fx, fy), (0, 255, 0), 2)
    cv2.imshow("Select Region", img)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cv2.destroyAllWindows() 