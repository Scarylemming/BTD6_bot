import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_best_matches(img1, img2, num_matches=3, threshold=0.5):
    """
    Find the best matches of img1 in img2 using template matching.
    
    Args:
        img1: Template image to find
        img2: Image to search in
        num_matches: Number of best matches to return
        threshold: Minimum confidence threshold for matches
    
    Returns:
        List of tuples: (confidence, (x, y), (width, height))
    """
    # Convert images to grayscale if they aren't already
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Perform template matching
    result = cv2.matchTemplate(img2, img1, cv2.TM_CCOEFF_NORMED)
    
    # Get template dimensions
    h, w = img1.shape
    
    # Find all locations where the correlation exceeds threshold
    locations = np.where(result >= threshold)
    matches = []
    
    for pt in zip(*locations[::-1]):  # Switch columns and rows
        confidence = result[pt[1], pt[0]]
        matches.append((confidence, pt, (w, h)))
    
    # Sort matches by confidence (descending)
    matches.sort(key=lambda x: x[0], reverse=True)
    
    # Remove overlapping matches (non-maximum suppression)
    filtered_matches = []
    for match in matches:
        confidence, (x, y), (w, h) = match
        
        # Check if this match overlaps significantly with any existing match
        overlap = False
        for existing_match in filtered_matches:
            ex, ey, ew, eh = existing_match[1][0], existing_match[1][1], existing_match[2][0], existing_match[2][1]
            
            # Calculate overlap area
            x_overlap = max(0, min(x + w, ex + ew) - max(x, ex))
            y_overlap = max(0, min(y + h, ey + eh) - max(y, ey))
            overlap_area = x_overlap * y_overlap
            
            # Calculate areas
            area1 = w * h
            area2 = ew * eh
            min_area = min(area1, area2)
            
            # If overlap is more than 50% of the smaller area, consider it overlapping
            if overlap_area > 0.5 * min_area:
                overlap = True
                break
        
        if not overlap:
            filtered_matches.append(match)
            if len(filtered_matches) >= num_matches:
                break
    
    return filtered_matches[:num_matches]

def visualize_matches(img1, img2, matches):
    """
    Visualize the matches found in the image.
    
    Args:
        img1: Template image
        img2: Image to search in
        matches: List of matches from find_best_matches
    """
    # Create a copy for visualization
    if len(img2.shape) == 3:
        display_img = img2.copy()
    else:
        display_img = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Draw rectangles around matches
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Blue, Red
    
    for i, (confidence, (x, y), (w, h)) in enumerate(matches):
        color = colors[i % len(colors)]
        cv2.rectangle(display_img, (x, y), (x + w, y + h), color, 2)
        
        # Add confidence text
        cv2.putText(display_img, f"{confidence:.2f}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Display the result
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Best {len(matches)} matches found")
    plt.axis('off')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load images
    img1 = cv2.imread("images/collection_event/tier2.png")
    img2 = cv2.imread("images/screenshots/2.png")
    
    if img1 is None or img2 is None:
        print("Error: Could not load images")
    else:
        # Find best 3 matches
        matches = find_best_matches(img1, img2, num_matches=3, threshold=0.3)
        
        print(f"Found {len(matches)} matches:")
        for i, (confidence, (x, y), (w, h)) in enumerate(matches):
            print(f"Match {i+1}: Confidence={confidence:.3f}, Position=({x}, {y}), Size=({w}, {h})")
        
        # Visualize the matches
        visualize_matches(img1, img2, matches) 