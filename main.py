import time
import pyautogui
import cv2
import numpy as np
from PIL import Image
import logging
import subprocess
import os
import psutil
from config import *
import json
from gameplay import Gameplay
import keyboard
import pytesseract
import difflib
from sentence_transformers import SentenceTransformer, util
from difflib import SequenceMatcher
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)

class BTD6Bot:
    def __init__(self):
        self.running = True
        self.setup_safety_features()
        self.MAPS = self.read_maps()


    def initialize_model(self):
        # Initialize the semantic model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Pre-compute embeddings for all map names
        self.map_embeddings = self.model.encode(self.MAPS)
        
    def setup_safety_features(self):
        """Setup emergency stop and other safety features"""
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = WAIT_TIME

    def is_game_running(self):
        """Check if BTD6 is currently running"""
        for proc in psutil.process_iter(['name']):
            try:
                if 'BloonsTD6' in proc.info['name']:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return False

    def launch_game(self):
        """Launch BTD6 if it's not already running"""
        try:
            if self.is_game_running():
                logging.info("BTD6 is already running")
                # Focus the BTD6 window
                try:
                    import pygetwindow as gw
                    windows = gw.getWindowsWithTitle(GAME_WINDOW_TITLE)
                    if windows:
                        windows[0].activate()
                        logging.info("Focused BTD6 window.")
                    else:
                        logging.warning("BTD6 window not found to focus.")
                except Exception as e:
                    logging.warning(f"Could not focus BTD6 window: {e}")
                return True
            if not os.path.exists(GAME_PATH):
                logging.error(f"Game not found at path: {GAME_PATH}")
                return False
            logging.info("Launching BTD6...")
            subprocess.Popen([GAME_PATH])
            for _ in range(30):
                if self.is_game_running():
                    logging.info("BTD6 launched successfully")
                    # time.sleep(5)
                    return True
                time.sleep(1)
            logging.error("Failed to launch BTD6")
            return False
        except Exception as e:
            logging.error(f"Error launching game: {e}")
            return False

    def find_image_on_screen(self, image_path, confidence=CONFIDENCE_THRESHOLD):
        """Find an image on the screen using template matching"""
        try:
            # Take screenshot
            screen = pyautogui.screenshot(region=SCREENSHOT_REGION)
            screen_np = np.array(screen)
            
            # Convert to BGR (OpenCV format)
            screen_np = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)
            
            # Load template
            template = cv2.imread(image_path)
            if template is None:
                logging.error(f"Template image not found: {image_path}")
                return None, None
                
            # Get template dimensions
            th, tw = template.shape[:2]
            
            # Perform template matching
            result = cv2.matchTemplate(screen_np, template, cv2.TM_CCOEFF_NORMED)
            
            # Find the best match
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Debug information
            logging.debug(f"Template matching for {image_path}:")
            logging.debug(f"Max confidence: {max_val:.3f}")
            logging.debug(f"Min confidence: {min_val:.3f}")
            
            # Save debug visualization if confidence is low
            if max_val < confidence:
                debug_img = screen_np.copy()
                cv2.rectangle(debug_img, max_loc, (max_loc[0] + tw, max_loc[1] + th), (0, 255, 0), 2)
                debug_path = f"debug_{os.path.basename(image_path)}"
                cv2.imwrite(debug_path, debug_img)
                logging.debug(f"Debug image saved to {debug_path}")
            
            if max_val >= confidence:
                center_x = max_loc[0] + tw // 2
                center_y = max_loc[1] + th // 2
                return (center_x, center_y), max_val
                
            logging.warning(f"Image not found with sufficient confidence. Max confidence: {max_val:.3f}")
            return None, max_val
            
        except Exception as e:
            logging.error(f"Error in find_image_on_screen: {e}")
            return None, None

    def click_image(self, image_path, confidence=CONFIDENCE_THRESHOLD):
        """Click on an image if found on screen"""
        location, max_val = self.find_image_on_screen(image_path, confidence)
        if location:
            pyautogui.click(location)
            return True
        return False

    def wait_and_click(self, image_path, description, confidence=CONFIDENCE_THRESHOLD):
        logging.info(f"Looking for {description}...")
        while self.running:
            if self.click_image(image_path, confidence):
                logging.info(f"Clicked {description}.")
                # time.sleep(1)
                return True
            # time.sleep(1)
        return False

    def read_maps(self):
        """Read and return a list of unique map names from placements.json"""
        try:
            with open('placements.json', 'r') as f:
                placements = json.load(f)
            
            # Extract unique map names using a set
            map_names = set()
            for placement in placements:
                if "map" in placement:
                    map_names.add(placement["map"])
            
            # Convert set to sorted list for consistent ordering
            return sorted(list(map_names))
        except Exception as e:
            logging.error(f"Error reading maps from placements.json: {e}")
            return []

    def find_best_matches(self, img1, img2, num_matches=3, threshold=0.5):
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

    def find_collectible_towers(self, tower_type="common", num_matches=3, threshold=0.3):
        """
        Find collectible towers of a specific type on the screen.
        
        Args:
            tower_type: Type of tower to find ("common", "tier2", "tier3", "tier4")
            num_matches: Number of best matches to return
            threshold: Minimum confidence threshold for matches
        
        Returns:
            List of tuples: (confidence, (center_x, center_y))
        """
        try:
            # Take screenshot
            screen = pyautogui.screenshot(region=SCREENSHOT_REGION)
            screen_np = np.array(screen)
            screen_np = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'images/screenshots/screen_{time.time()}.png', screen_np)
            
            # Load template image
            template_path = f'images/collection_event/{tower_type}.png'
            template = cv2.imread(template_path)
            if template is None:
                logging.error(f"Template image not found: {template_path}")
                return []
            
            # Find matches
            matches = self.find_best_matches(template, screen_np, num_matches, threshold)
            
            # Convert to center coordinates
            tower_positions = []
            for confidence, (x, y), (w, h) in matches:
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Add debugging information
                logging.info(f"Template match: confidence={confidence:.3f}, raw_coords=({x}, {y}), size=({w}, {h}), center=({center_x}, {center_y})")
                
                tower_positions.append((confidence, (center_x, center_y)))
            
            logging.info(f"Found {len(tower_positions)} {tower_type} towers")
            return tower_positions
            
        except Exception as e:
            logging.error(f"Error finding {tower_type} towers: {e}")
            return []

    def collection_event(self):
        """Handle collection event using template matching instead of hardcoded coordinates"""
        if self.find_image_on_screen('images/collection_event/collect.png')[0] is not None:
            print("Collect towers")
            self.wait_and_click('images/collection_event/collect.png', 'Click Collection Event')
            time.sleep(1)
            
            # Define tower types to collect (in order of priority)
            tower_types = ["tier2"]
            
            for tower_type in tower_types:
                print(tower_type)
                # Find collectible towers of this type
                tower_positions = self.find_collectible_towers(tower_type, num_matches=3, threshold=0.8)
                
                if tower_positions:
                    logging.info(f"Collecting {len(tower_positions)} {tower_type} towers")
                    
                    for confidence, (center_x, center_y) in tower_positions:
                        # Add debugging information
                        logging.info(f"Attempting to click {tower_type} tower at ({center_x}, {center_y}) with confidence {confidence:.3f}")
                        
                        # Click on the tower
                        pyautogui.click(center_x, center_y)
                        time.sleep(1)  # Increased delay for better reliability
                        
                        # Click in the middle of the screen to close any popup
                        pyautogui.click(1000, 1000)
                        time.sleep(0.5)
                        
                        logging.info(f"Collected {tower_type} tower at ({center_x}, {center_y}) with confidence {confidence:.3f}")
                else:
                    logging.warning(f"No {tower_type} towers found to collect. Max confidence: {max([m[0] for m in self.find_best_matches(template, screen_np, 1, 0.1)]) if 'template' in locals() and 'screen_np' in locals() else 'unknown'}")
            
            # Continue with the collection event
            self.wait_and_click('images/collection_event/continue.png', 'Back to collection event')
            
            # Check if we need to go back to collection event screen
            if (self.find_image_on_screen('images/collection_event/collection_event.png')[0] is not None and 
                self.find_image_on_screen('images/collection_event/collect.png')[0] is None):
                pyautogui.click(84, 67)
            
            self.wait_and_click('images/collection_event/back.png', 'Back to home')
        return True

    def place_towers_for_map(self, map_name):
        """Place towers for a specific map based on the placements.json configuration"""
        try:
            with open('placements.json', 'r') as f:
                all_placements = json.load(f)
            
            # Filter placements for the current map
            map_placements = [p for p in all_placements if p.get("map") == map_name]
            
            if not map_placements:
                logging.warning(f"No tower placements found for map: {map_name}")
                return False

            gameplay = Gameplay()
            for placement in map_placements:
                try:
                    method = placement.get("method", "key")
                    coordinates = tuple(placement["coordinates"])
                    if method == "key":
                        key = placement["key"]
                        gameplay.place_tower(key, coordinates)
                    elif method == "image":
                        if "image" not in placement:
                            logging.error(f"Missing 'image' field in placement: {placement}")
                            continue
                        image = placement["image"]
                        gameplay.place_tower_by_image(image, coordinates)
                    else:
                        logging.warning(f"Unknown placement method: {method} in placement: {placement}")
                    time.sleep(0.5)
                except Exception as e:
                    logging.error(f"Error placing tower for placement {placement}: {e}")
            logging.info(f"All towers placed for map: {map_name}")
            return True
        except Exception as e:
            logging.error(f"Error loading or processing placements.json: {e}")
            return False

    def find_semantic_match(self, text, threshold=0.2):
        """Find the most semantically similar map name using sentence transformers"""
        try:
            # Encode the input text
            text_embedding = self.model.encode([text])[0]
            
            # Calculate cosine similarity with all map names
            similarities = util.cos_sim(text_embedding, self.map_embeddings)[0]
            
            # Get the best match
            best_idx = similarities.argmax()
            best_score = similarities[best_idx].item()
            
            logging.info(f"Text to match: {text}")
            logging.info(f"Best match: {self.MAPS[best_idx]} with score: {best_score:.3f}")
            
            # Return the best match if it's above threshold
            if best_score >= threshold:
                return self.MAPS[best_idx]
            return None
            
        except Exception as e:
            logging.error(f"Error in semantic matching: {e}")
            return None

    def clean_text(self, text):
        # Lowercase, remove non-alphanumeric, strip
        import re
        return re.sub(r'[^a-z0-9 ]', '', text.lower().strip())

    def find_best_map_match(self, ocr_text, fuzzy_threshold=0.7, semantic_threshold=0.2):
        cleaned_ocr = self.clean_text(ocr_text)
        cleaned_maps = [self.clean_text(m) for m in self.MAPS]

        # Fuzzy matching
        best_fuzzy_score = 0
        best_fuzzy_map = None
        for map_name, cleaned_map in zip(self.MAPS, cleaned_maps):
            score = SequenceMatcher(None, cleaned_ocr, cleaned_map).ratio()
            if score > best_fuzzy_score:
                best_fuzzy_score = score
                best_fuzzy_map = map_name

        logging.info(f"Best fuzzy match: {best_fuzzy_map} (score: {best_fuzzy_score:.2f})")

        if best_fuzzy_score >= fuzzy_threshold:
            return best_fuzzy_map

        # Fall back to semantic matching
        best_semantic_map = self.find_semantic_match(ocr_text, threshold=semantic_threshold)
        logging.info(f"Best semantic match: {best_semantic_map}")

        return best_semantic_map

    def read_which_map_to_play(self):
        # Take a screenshot at the coordinates x=304, y=218, width=366, height=28
        screenshot = pyautogui.screenshot(region=(304, 215, 366, 45))
        screenshot.save('images/misc/map_to_play.png')
        
        # Convert to grayscale for better OCR
        screenshot_np = np.array(screenshot)
        gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)
        
        # Apply thresholding to make text more clear
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use OCR with optimized parameters
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        text = pytesseract.image_to_string(
            thresh,
            config=custom_config,
            lang='eng'
        )
        
        # Clean the text
        text = text.strip().lower()
        logging.info(f"OCR raw text: {text}")
        
        if text == "" : 
            closest_match = "cubism"
            print("We got no text read my OCR, but we will play cubism because that's the only problematic map.")
        else :
            # Find the closest match using semantic similarity
            closest_match = self.find_best_map_match(text)
        print("Map to play:", closest_match)
        return closest_match
    
    def play_map(self, map_name):
        # Click on the map
        pyautogui.click(500,400)
        # self.wait_and_click(f'images/maps/{map_name}.png', f'Map: {map_name}')
        # Click on the difficulty
        self.wait_and_click(f'images/difficulty/Easy.png', f'Easy difficulty')
        # Click on the mode
        self.wait_and_click(f'images/difficulty/{MODE}.png', f'Mode: {MODE}')
        # Click on the round
        self.wait_and_click(f'images/misc/round.png', 'Find round to start placing towers')
        # Place towers
        self.place_towers_for_map(map_name)
        # Press space to start the game
        keyboard.press_and_release('space')
        time.sleep(0.1)
        keyboard.press_and_release('space')
        # Click on the next
        self.wait_and_click(f'images/end/next.png', 'Finished, click next')

    def take_screenshot_and_compare(self):
        # Take a screenshot of the map
        screenshot = pyautogui.screenshot(region=(290, 220, 437, 244))
        screenshot.save(f'images/maps/{time.time()}.png')
        screenshot_np = np.array(screenshot.convert('RGB'))
        screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)

        best_score = -1
        best_file = None

        # Take only files without numbers in the name
        files = [file for file in os.listdir('images/maps') if not any(char.isdigit() for char in file)]

        for file in files:
            if file.endswith('.png'):
                screenshot_to_compare = Image.open(f'images/maps/{file}').convert('RGB')
                compare_np = np.array(screenshot_to_compare)
                compare_gray = cv2.cvtColor(compare_np, cv2.COLOR_RGB2GRAY)

                # Resize if needed
                if screenshot_gray.shape != compare_gray.shape:
                    compare_gray = cv2.resize(compare_gray, (screenshot_gray.shape[1], screenshot_gray.shape[0]))

                score, _ = ssim(screenshot_gray, compare_gray, full=True)
                if score > best_score:
                    best_score = score
                    best_file = file

        print(f"Best match: {best_file} (SSIM: {best_score:.3f})")
        return best_file.split('.')[0]

    def debug_template_matching(self, tower_type="tier2", threshold=0.3):
        """
        Debug method to visualize template matching results and save the image.
        """
        try:
            # Take screenshot
            screen = pyautogui.screenshot(region=SCREENSHOT_REGION)
            screen_np = np.array(screen)
            screen_np = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)
            
            # Load template image
            template_path = f'images/collection_event/{tower_type}.png'
            template = cv2.imread(template_path)
            if template is None:
                logging.error(f"Template image not found: {template_path}")
                return
            
            # Find matches
            matches = self.find_best_matches(template, screen_np, num_matches=3, threshold=threshold)
            
            # Create visualization
            display_img = screen_np.copy()
            
            for i, (confidence, (x, y), (w, h)) in enumerate(matches):
                # Draw rectangle around match
                color = [(0, 255, 0), (255, 0, 0), (0, 0, 255)][i % 3]  # Green, Blue, Red
                cv2.rectangle(display_img, (x, y), (x + w, y + h), color, 2)
                
                # Draw center point
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.circle(display_img, (center_x, center_y), 5, color, -1)
                
                # Add confidence text
                cv2.putText(display_img, f"{confidence:.2f}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                logging.info(f"Match {i+1}: confidence={confidence:.3f}, coords=({x}, {y}), center=({center_x}, {center_y})")
            
            # Save the debug image
            debug_path = f'debug_{tower_type}_matching.png'
            cv2.imwrite(debug_path, display_img)
            logging.info(f"Debug image saved as: {debug_path}")
            
            return matches
            
        except Exception as e:
            logging.error(f"Error in debug_template_matching: {e}")
            return []

    def run(self):
        """Main bot loop"""
        # Step 1: Click Event Collection
        self.wait_and_click('images/misc/collect_event.png', 'Collect Event button')
        # Step 1.5 : Click Play
        self.wait_and_click('images/misc/play_button.png', 'Play button')
        # Step 2 : Read the map to play
        # map_name = self.read_which_map_to_play()
        # Step 2.5 : Take screenshot of the map
        map_name = self.take_screenshot_and_compare()
        # Step 3 : Play the map
        self.play_map(map_name)
        # Step 5 : Click Home 
        self.wait_and_click('images/end/home.png', 'Click Home')
        time.sleep(10)
        # Step 6 : Check Collect Event
        self.collection_event()

if __name__ == "__main__":
    bot = BTD6Bot()
    if not bot.launch_game():
        logging.error("Failed to launch game. Exiting...")
    # bot.initialize_model()
    while True : 
        bot.run() 