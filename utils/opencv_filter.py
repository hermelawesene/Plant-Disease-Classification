import cv2
import numpy as np

def is_leaf(image: np.ndarray) -> bool:
    # Resize image to a standard size for consistency (e.g., 256x256 as in your model)
    image = cv2.resize(image, (256, 256))
    
    # Convert to HSV and apply Gaussian blur to reduce noise
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
    
    # Expanded HSV range to include varied green shades and diseased leaves
    lower_green = np.array([20, 30, 30])  # Slightly broader range
    upper_green = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Optional: Morphological operations to clean mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    # Relative area threshold based on image size (e.g., 2% of total area)
    min_area_threshold = 0.02 * 256 * 256  # ~1310 pixels for 256x256
    
    if area < min_area_threshold:
        return False
    
    # Compute solidity
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    # Relaxed solidity range to accommodate irregular leaves
    return (area > min_area_threshold) and (0.6 < solidity < 0.98)