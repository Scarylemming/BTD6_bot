import time
import pyautogui
import cv2
import numpy as np
from PIL import Image
import logging
import subprocess
import os
import psutil
import json
from gameplay import Gameplay
import keyboard
import pytesseract
import difflib
from difflib import SequenceMatcher
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import sys
import importlib.util

# Command line argument handling
def load_config(config_file=None):
    """Load configuration from specified file or use default"""
    if config_file is None:
        # Default config
        import config
        return config
    else:
        # Load specified config file
        try:
            # Remove .py extension if present
            if config_file.endswith('.py'):
                config_file = config_file[:-3]
            
            # Import the config module
            spec = importlib.util.spec_from_file_location("config", f"{config_file}.py")
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            print(f"✅ Loaded config from: {config_file}.py")
            return config_module
        except Exception as e:
            print(f"❌ Error loading config file '{config_file}.py': {e}")
            print("🔄 Falling back to default config...")
            import config
            return config

# Parse command line arguments
def parse_arguments():
    """Parse command line arguments"""
    config_file = None
    
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        print(f"🎯 Using config file: {config_file}")
    
    return config_file

# Load configuration
CONFIG_FILE = parse_arguments()
CONFIG = load_config(CONFIG_FILE)

# Import all config variables
GAME_WINDOW_TITLE = CONFIG.GAME_WINDOW_TITLE
GAME_WINDOW_SIZE = CONFIG.GAME_WINDOW_SIZE
RUN_IN_BACKGROUND = CONFIG.RUN_IN_BACKGROUND
MINIMIZE_AFTER_LAUNCH = CONFIG.MINIMIZE_AFTER_LAUNCH
RESTORE_WINDOW_BEFORE_ACTION = CONFIG.RESTORE_WINDOW_BEFORE_ACTION
MINIMIZE_AFTER_ACTION = CONFIG.MINIMIZE_AFTER_ACTION
GAME_PATH = CONFIG.GAME_PATH
MAP_DIFFICULTY = CONFIG.MAP_DIFFICULTY
DEFAULT_MAP = CONFIG.DEFAULT_MAP
GAME_DIFFICULTY = CONFIG.GAME_DIFFICULTY
MODE = CONFIG.MODE
DEFAULT_TOWERS = CONFIG.DEFAULT_TOWERS
ROUNDS_TO_PLAY = CONFIG.ROUNDS_TO_PLAY
WAIT_TIME = CONFIG.WAIT_TIME
CONFIDENCE_THRESHOLD = CONFIG.CONFIDENCE_THRESHOLD
SCREENSHOT_REGION = CONFIG.SCREENSHOT_REGION
EMERGENCY_STOP_REGION = CONFIG.EMERGENCY_STOP_REGION

# Tesseract configuration
TESSERACT_PATH = getattr(CONFIG, 'TESSERACT_PATH', None)
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    print(f"✅ Using Tesseract from: {TESSERACT_PATH}")
else:
    print("ℹ️ Using default Tesseract installation")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class BTD6Bot:
    def __init__(self):
        print("🚀 Initializing BTD6 Bot...")
        self.running = True
        
        print("🔧 Setting up safety features...")
        self.setup_safety_features()
        
        print("🔍 Checking Tesseract installation...")
        self.check_tesseract_installation()
        
        print("🗺️ Loading map configurations...")
        self.MAPS = self.get_maps()
        print(f"✅ Loaded {len(self.MAPS)} maps")
        self.game_won = False

    def setup_safety_features(self):
        """Setup emergency stop and other safety features"""
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = WAIT_TIME

    def check_tesseract_installation(self):
        """Check if Tesseract is properly installed and accessible"""
        try:
            # Try to get Tesseract version
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract version: {version}")
            return True
        except Exception as e:
            print("❌ Tesseract is not installed or not found in PATH!")
            print("\n📋 To fix this issue:")
            print("1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
            print("2. Install it and make sure to check 'Add to PATH' during installation")
            print("3. Or specify the path in your config file:")
            print("   TESSERACT_PATH = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'")
            print("\n🔧 Alternative solutions:")
            print("- Restart your terminal/IDE after installation")
            print("- Add Tesseract installation directory to your system PATH")
            print("- Use the TESSERACT_PATH configuration option")
            
            # Check if we have a custom path configured
            if TESSERACT_PATH:
                print(f"\n🔍 Trying configured path: {TESSERACT_PATH}")
                try:
                    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
                    version = pytesseract.get_tesseract_version()
                    print(f"✅ Tesseract found at configured path: {version}")
                    return True
                except Exception as e2:
                    print(f"❌ Tesseract not found at configured path: {e2}")
            
            print(f"\n❌ Error details: {e}")
            return False

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
            print("🎮 Checking if BTD6 is running...")
            if self.is_game_running():
                print("✅ BTD6 is already running")
                logging.info("BTD6 is already running")
                # Focus the BTD6 window
                print("🎯 Focusing BTD6 window...")
                try:
                    import pygetwindow as gw
                    windows = gw.getWindowsWithTitle(GAME_WINDOW_TITLE)
                    if windows:
                        windows[0].activate()
                        print("✅ BTD6 window focused")
                        logging.info("Focused BTD6 window.")
                    else:
                        print("⚠️ BTD6 window not found to focus")
                        logging.warning("BTD6 window not found to focus.")
                except Exception as e:
                    print(f"⚠️ Could not focus BTD6 window: {e}")
                    logging.warning(f"Could not focus BTD6 window: {e}")
                return True
            if not os.path.exists(GAME_PATH):
                print(f"❌ Game not found at path: {GAME_PATH}")
                logging.error(f"Game not found at path: {GAME_PATH}")
                return False
            print("🚀 Launching BTD6...")
            logging.info("Launching BTD6...")
            subprocess.Popen([GAME_PATH])
            print("⏳ Waiting for BTD6 to start...")
            for i in range(30):
                if self.is_game_running():
                    print("✅ BTD6 launched successfully")
                    logging.info("BTD6 launched successfully")
                    # time.sleep(5)
                    return True
                time.sleep(1)
                if i % 5 == 0:  # Show progress every 5 seconds
                    print(f"⏳ Still waiting... ({i+1}/30 seconds)")
            print("❌ Failed to launch BTD6")
            logging.error("Failed to launch BTD6")
            return False
        except Exception as e:
            print(f"❌ Error launching game: {e}")
            logging.error(f"Error launching game: {e}")
            return False

    def find_image_on_screen(self, image_path, confidence=CONFIDENCE_THRESHOLD, image_name=""):
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
                
            logging.warning(f"Image {image_name} not found with sufficient confidence. Max confidence: {max_val:.3f}")
            return None, max_val
            
        except Exception as e:
            logging.error(f"Error in find_image_on_screen for {image_name}: {e}")
            return None, None

    def click_image(self, image_path, confidence=CONFIDENCE_THRESHOLD, image_name=""):
        """Click on an image if found on screen"""
        location, max_val = self.find_image_on_screen(image_path, confidence, image_name)
        if location:
            pyautogui.click(location)
            return True
        return False

    def wait_and_click(self, image_path, description, confidence=CONFIDENCE_THRESHOLD, image_name=""):
        logging.info(f"Looking for {description}...")
        while self.running:
            if self.click_image(image_path, confidence, image_name):
                logging.info(f"Clicked {description}.")
                # time.sleep(1)
                return True
            # time.sleep(1)
        return False


    def wait_for_any_image(self, image_paths, confidence=0.8, image_name=""):
        logging.info(f"Waiting for any of these images: {image_paths}")
        while True:
            for image_path in image_paths:
                if self.find_image_on_screen(image_path, confidence, image_name)[0] is not None:
                    logging.info(f"Found image: {image_path}")
                    return True
            time.sleep(1)
        return False

    def read_maps(self):
        """Read and return a list of unique map names from placements.json"""
        try:
            with open('placements.json', 'r') as f:
                placements = json.load(f)
            
            # Extract unique map names using a set
            map_names = set()
            for map_name in placements.keys():
                map_names.add(map_name)
            
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
        if self.find_image_on_screen('images/collection_event/collect.png', image_name="collect")[0] is not None:
            print("Collect towers")
            self.wait_and_click('images/collection_event/collect.png', 'Click Collection Event')
            self.wait_for_any_image(['images/collection_event/tier4.png', 'images/collection_event/tier3.png', 'images/collection_event/tier2.png', 'images/collection_event/tier1.png'])
            
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
            if (self.find_image_on_screen('images/collection_event/collection_event.png', image_name="collection_event")[0] is not None and 
                self.find_image_on_screen('images/collection_event/collect.png', image_name="collect")[0] is None):
                pyautogui.click(84, 67)
            
            self.wait_and_click('images/collection_event/back.png', 'Back to home')
        return True

    def place_towers_for_round(self, map_name, current_round, game_mode=None):
        """Place towers for a specific round during gameplay"""
        if game_mode is None:
            game_mode = GAME_DIFFICULTY.lower()
            
        try:
            with open('placements.json', 'r') as f:
                all_placements = json.load(f)
            
            # Check if map exists
            if map_name not in all_placements:
                logging.warning(f"Map '{map_name}' not found in placements.json")
                return False
            
            map_data = all_placements[map_name]
            
            # Check if game mode exists for this map
            if game_mode not in map_data:
                logging.warning(f"Game mode '{game_mode}' not found for map '{map_name}'")
                return False
            
            game_mode_data = map_data[game_mode]
            
            # Check if mode exists for this game mode
            mode = MODE.lower()
            if mode not in game_mode_data:
                logging.warning(f"Mode '{mode}' not found for map '{map_name}' in '{game_mode}' mode")
                return False
            
            mode_data = game_mode_data[mode]
            rounds_data = mode_data.get("rounds", {})
            
            # Check if there are actions for the current round
            if str(current_round) not in rounds_data:
                logging.debug(f"No tower placements for round {current_round} on map '{map_name}' in '{game_mode}' mode with '{mode}' mode")
                return True  # Not an error, just no placements for this round
            
            round_actions = rounds_data[str(current_round)]
            gameplay = Gameplay()
            
            logging.info(f"Processing round {current_round} for map '{map_name}' in '{game_mode}' mode with '{mode}' mode")
            
            for action in round_actions:
                try:
                    method = action.get("method", "key")
                    coordinates = tuple(action["coordinates"])
                    action_type = action.get("action", "place")
                    
                    if action_type == "place":
                        if method == "key":
                            key = action["key"]
                            gameplay.place_tower(key, coordinates)
                            logging.info(f"Placed tower with key '{key}' at {coordinates}")
                        elif method == "image":
                            if "image" not in action:
                                logging.error(f"Missing 'image' field in action: {action}")
                                continue
                            image = action["image"]
                            gameplay.place_tower_by_image(image, coordinates)
                            logging.info(f"Placed tower with image '{image}' at {coordinates}")
                        else:
                            logging.warning(f"Unknown placement method: {method} in action: {action}")
                    
                    elif action_type == "upgrade":
                        if "upgrade_path" in action:
                            # Use the new upgrade_tower method with upgrade path
                            upgrade_path = action["upgrade_path"]
                            gameplay.upgrade_tower(coordinates, upgrade_path)
                            logging.info(f"Upgraded tower at {coordinates} with path {upgrade_path}")
                        elif method == "key":
                            key = action["key"]
                            # Click on the tower at coordinates, then press the upgrade key
                            pyautogui.click(coordinates)
                            time.sleep(0.1)
                            keyboard.press_and_release(key)
                            logging.info(f"Upgraded tower at {coordinates} with key '{key}'")
                        else:
                            logging.warning(f"Unknown upgrade method: {method} in action: {action}")
                    
                    elif action_type == "sell":
                        # Click on the tower at coordinates, then press the sell key
                        pyautogui.click(coordinates)
                        time.sleep(0.1)
                        keyboard.press_and_release('backspace')  # Default sell key
                        logging.info(f"Sold tower at {coordinates}")
                    
                    elif action_type == "target":
                        # Click on the tower at coordinates, then press the target key
                        pyautogui.click(coordinates)
                        time.sleep(0.1)
                        target_key = action.get("key", "tab")  # Default target key
                        keyboard.press_and_release(target_key)
                        logging.info(f"Changed target priority at {coordinates} with key '{target_key}'")
                    
                    else:
                        logging.warning(f"Unknown action type: {action_type} in action: {action}")
                    
                    time.sleep(0.5)  # Small delay between actions
                    
                except Exception as e:
                    logging.error(f"Error processing action {action}: {e}")
            
            logging.info(f"Completed round {current_round} actions for map '{map_name}' in '{game_mode}' mode with '{mode}' mode")
            return True
            
        except Exception as e:
            logging.error(f"Error loading or processing placements.json for round {current_round}: {e}")
            return False

    def place_towers_for_map(self, map_name, game_mode=None):
        """Place towers for a specific map and game mode based on the placements.json configuration"""
        if game_mode is None:
            game_mode = GAME_DIFFICULTY.lower()
            
        try:
            with open('placements.json', 'r') as f:
                all_placements = json.load(f)
            
            # Check if map exists
            if map_name not in all_placements:
                logging.warning(f"Map '{map_name}' not found in placements.json")
                return False
            
            map_data = all_placements[map_name]
            
            # Check if game mode exists for this map
            if game_mode not in map_data:
                logging.warning(f"Game mode '{game_mode}' not found for map '{map_name}'")
                return False
            
            game_mode_data = map_data[game_mode]
            
            # Check if mode exists for this game mode
            mode = MODE.lower()
            if mode not in game_mode_data:
                logging.warning(f"Mode '{mode}' not found for map '{map_name}' in '{game_mode}' mode")
                return False
            
            mode_data = game_mode_data[mode]
            rounds_data = mode_data.get("rounds", {})
            
            if not rounds_data:
                logging.warning(f"No rounds data found for map '{map_name}' in '{game_mode}' mode with '{mode}' mode")
                return False

            # Determine starting round based on difficulty and mode
            if game_mode == "hard" and mode == "impoppable":
                starting_round = 6  # Impoppable starts with round 6
                logging.info(f"Impoppable mode detected - placing only round {starting_round} towers")
            elif game_mode == "easy" and mode == "deflation" :
                starting_round = 30
                logging.info(f"Deflation mode detected - placing only round {starting_round} towers")
            else:
                starting_round = 1  # Easy Standard and others start with round 1
                logging.info(f"Standard mode detected - placing only round {starting_round} towers")

            gameplay = Gameplay()
            
            # Only place towers from the starting round
            starting_round_str = str(starting_round)
            if starting_round_str not in rounds_data:
                logging.warning(f"No actions found for starting round {starting_round}")
                return False
            
            round_actions = rounds_data[starting_round_str]
            
            logging.info(f"Processing starting round {starting_round} for map '{map_name}' in '{game_mode}' mode with '{mode}' mode")
            
            for action in round_actions:
                try:
                    method = action.get("method", "key")
                    coordinates = tuple(action["coordinates"])
                    action_type = action.get("action", "place")
                    print(action_type, action)
                    
                    if action_type == "place":
                        if method == "key":
                            key = action["key"]
                            gameplay.place_tower(key, coordinates)
                            logging.info(f"Placed tower with key '{key}' at {coordinates}")
                            time.sleep(0.2)
                        elif method == "image":
                            if "image" not in action:
                                logging.error(f"Missing 'image' field in action: {action}")
                                continue
                            image = action["image"]
                            gameplay.place_tower_by_image(image, coordinates)
                            logging.info(f"Placed tower with image '{image}' at {coordinates}")
                        else:
                            logging.warning(f"Unknown placement method: {method} in action: {action}")
                    
                    elif action_type == "upgrade":
                        if "upgrade_path" in action:
                            # Use the new upgrade_tower method with upgrade path
                            upgrade_path = action["upgrade_path"]
                            gameplay.upgrade_tower(coordinates, upgrade_path)
                            logging.info(f"Upgraded tower at {coordinates} with path {upgrade_path}")
                        elif method == "key":
                            key = action["key"]
                            # Click on the tower at coordinates, then press the upgrade key
                            pyautogui.click(coordinates)
                            time.sleep(0.1)
                            keyboard.press_and_release(key)
                            logging.info(f"Upgraded tower at {coordinates} with key '{key}'")
                        else:
                            logging.warning(f"Unknown upgrade method: {method} in action: {action}")
                    
                    elif action_type == "sell":
                        # Click on the tower at coordinates, then press the sell key
                        pyautogui.click(coordinates)
                        time.sleep(0.1)
                        keyboard.press_and_release('backspace')  # Default sell key
                        logging.info(f"Sold tower at {coordinates}")
                    
                    elif action_type == "target":
                        # Click on the tower at coordinates, then press the target key
                        pyautogui.click(coordinates)
                        time.sleep(0.1)
                        target_key = action.get("key", "tab")  # Default target key
                        keyboard.press_and_release(target_key)
                        logging.info(f"Changed target priority at {coordinates} with key '{target_key}'")
                    
                    else:
                        logging.warning(f"Unknown action type: {action_type} in action: {action}")
                    
                    time.sleep(0.5)  # Small delay between actions
                    
                except Exception as e:
                    logging.error(f"Error processing action {action}: {e}")
            
            logging.info(f"Starting round {starting_round} towers placed for map '{map_name}' in '{game_mode}' mode with '{mode}' mode")
            return True
            
        except Exception as e:
            logging.error(f"Error loading or processing placements.json: {e}")
            return False

    def clean_text(self, text):
        # Lowercase, remove non-alphanumeric, strip
        import re
        return re.sub(r'[^a-z0-9 ]', '', text.lower().strip())

    def find_best_map_match(self, ocr_text, fuzzy_threshold=0.7):
        cleaned_ocr = self.clean_text(ocr_text)
        print(f"Cleaned OCR: {cleaned_ocr}")
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

        # If no good match found, return the best match anyway
        logging.warning(f"No good match found, using best available: {best_fuzzy_map}")
        return best_fuzzy_map

    def read_which_map_to_play(self):
        # Take a screenshot at the coordinates x=304, y=218, width=366, height=28
        screenshot = pyautogui.screenshot(region=(304, 215, 366, 45))
        # screenshot.save('images/misc/map_to_play.png')
        
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
            # Find the closest match using fuzzy matching
            closest_match = self.find_best_map_match(text)
        print("Map to play:", closest_match)
        return closest_match

    def navigate_to_map(self, map_name):
        # Click on the map
        if MAP_DIFFICULTY == "beginner":
            pyautogui.click(500,400)
        elif MAP_DIFFICULTY == "intermediate":
            pyautogui.click(950,400)
        elif MAP_DIFFICULTY == "advanced":
            pyautogui.click(1400,400)
        elif MAP_DIFFICULTY == "expert":
            pyautogui.click(500,750)

        # Click on the difficulty
        self.wait_and_click(f'images/difficulty/{GAME_DIFFICULTY}.png', f'{GAME_DIFFICULTY} difficulty')
        # Click on the mode
        self.wait_and_click(f'images/difficulty/{MODE}.png', f'Mode: {MODE}')
        # If Impoppable, click on OK to pass the warning
        if MODE in ["impoppable", "deflation"]:
            self.wait_and_click(f'images/misc/ok.png', 'OK button')

    
    def play_map(self, map_name):
        # Click on the round
        self.wait_for_any_image([f'images/misc/churchill.png'])
        # Place initial towers - uses GAME_DIFFICULTY automatically
        self.place_towers_for_map(map_name)
        # Press space to start the game
        keyboard.press_and_release('space')
        time.sleep(0.1)
        keyboard.press_and_release('space')
        
        # For easy mode, just wait for game to finish without monitoring rounds
        if GAME_DIFFICULTY.lower() == "easy" and MODE.lower() != "deflation":
            logging.info("Easy mode detected - skipping round monitoring")
            # Wait for game to finish by checking for victory/next screen
            while self.running:
                if self.is_game_finished():
                    logging.info("Game finished detected in easy mode")
                    break
                time.sleep(1)
        elif MODE.lower() == "deflation": # If Deflation, click on the next button, then on the freeplay mode, then on the ok button, then on the space button, then on the insta monkey button
            logging.info("Deflation mode detected - skipping round monitoring")
            # Wait for game to finish by checking for victory/next screen
            while self.running:
                if self.is_game_finished():
                    logging.info("Game finished detected in deflation mode")
                    break
                time.sleep(1)
            self.wait_and_click(f'images/end/next.png', 'Finished, click next')
            self.wait_and_click(f'images/misc/play_freeplay.png', 'Click on the freeplay mode', confidence=0.8)
            self.wait_for_any_image([f'images/misc/freeplay.png'])
            self.wait_and_click(f'images/misc/ok.png', 'Click on the freeplay mode')
            time.sleep(1)
            keyboard.press_and_release('space')
            self.wait_and_click(f'images/misc/insta_monkey.png', 'Click on the insta monkey')
        else:
            # Monitor rounds and execute actions during gameplay for other difficulties
            self.monitor_rounds_and_execute(map_name)

    def restart_game(self): # To prevent lag from accumulating RAM
        keyboard.press_and_release('esc')
        self.wait_and_click(f'images/misc/quit.png', 'Quit game')

        time.sleep(10)

        # Launch the game back
        self.launch_game()
        self.finish_launching_game()

    def finish_launching_game(self):
        self.wait_and_click(f'images/misc/start.png', 'Start BTD6')
        self.wait_for_any_image([f'images/misc/modded_client.png'])
        time.sleep(1)
        self.wait_and_click(f'images/misc/continue.png', 'Continue in the Modder warning')

    def take_screenshot_and_compare(self, save_screenshot=False):
        # Take a screenshot of the map
        if MAP_DIFFICULTY == "beginner":
            screenshot = pyautogui.screenshot(region=(290, 220, 437, 244))
        elif MAP_DIFFICULTY == "intermediate":
            screenshot = pyautogui.screenshot(region=(755, 217, 407, 270))
        elif MAP_DIFFICULTY == "advanced":
            screenshot = pyautogui.screenshot(region=(1225, 219, 408, 262))
        elif MAP_DIFFICULTY == "expert":
            screenshot = pyautogui.screenshot(region=(288, 565, 398, 269))
        
        # Save screenshot with descriptive name for debugging
        timestamp = int(time.time())
        debug_filename = f'debug_map_screenshot_{timestamp}.png'
        debug_path = f'debug_images/{debug_filename}'
        screenshot.save(debug_path)
        logging.info(f"Map screenshot saved for debugging: {debug_path}")
        
        # Also save with timestamp for historical reference
        if save_screenshot:
            timestamp_filename = f'{timestamp}.png'
            timestamp_path = f'images/maps/{MAP_DIFFICULTY}/{timestamp_filename}'
            screenshot.save(timestamp_path)
        
        screenshot_np = np.array(screenshot.convert('RGB'))
        screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)

        best_score = -1
        best_file = None
        all_scores = []  # Store all scores for debugging

        # Take only files without numbers in the name
        files = [file for file in os.listdir(f'images/maps/{MAP_DIFFICULTY}') if not any(char.isdigit() for char in file)]
        logging.info(f"Comparing against {len(files)} map template files")

        for file in files:
            if file.endswith('.png'):
                try:
                    screenshot_to_compare = Image.open(f'images/maps/{MAP_DIFFICULTY}/{file}').convert('RGB')
                    compare_np = np.array(screenshot_to_compare)
                    compare_gray = cv2.cvtColor(compare_np, cv2.COLOR_RGB2GRAY)

                    # Resize if needed
                    if screenshot_gray.shape != compare_gray.shape:
                        compare_gray = cv2.resize(compare_gray, (screenshot_gray.shape[1], screenshot_gray.shape[0]))
                        logging.debug(f"Resized {file} to match screenshot dimensions")

                    score, _ = ssim(screenshot_gray, compare_gray, full=True)
                    all_scores.append((file, score))
                    
                    if score > best_score:
                        best_score = score
                        best_file = file
                        
                except Exception as e:
                    logging.warning(f"Error processing {file}: {e}")

        # Log all scores for debugging (top 5)
        all_scores.sort(key=lambda x: x[1], reverse=True)
        logging.info("Top 5 map matches:")
        for i, (file, score) in enumerate(all_scores[:5]):
            logging.info(f"  {i+1}. {file}: {score:.3f}")

        print(f"Best match: {best_file} (SSIM: {best_score:.3f})")
        logging.info(f"Selected map: {best_file} with confidence {best_score:.3f}")
        
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

    def get_maps(self):
        """Get map names - either from hardcoded list or JSON file"""
        # Hardcoded map list for faster startup
        # This avoids the need to parse the JSON file during initialization
        hardcoded_maps = [
            "in_the_loop", "scrapyard", "tree_stump", "winter_park", "the_cabin",
            "spa_pits", "carved", "town_center", "end_of_the_road", "monkey_meadow",
            "cubism", "skates", "tinkerton", "middle_of_the_road", "one_two_tree",
            "resort", "lotus_island", "candy_falls", "park_path", "alpine_run",
            "frozen_over", "four_circles", "hedge", "logs"
        ]
        
        # To use JSON loading instead (slower but more flexible), uncomment the line below:
        # return self.read_maps()
        
        return hardcoded_maps

    def read_current_round(self):
        """Read the current round number from the game screen - optimized for dataset collection"""
        try:
            # Take a screenshot of the round number area
            screenshot = pyautogui.screenshot(region=(1365, 80, 76, 42))
            
            # Convert to grayscale for better OCR
            screenshot_np = np.array(screenshot)
            gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)
            
            # Use only the best preprocessing technique (Otsu thresholding)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Use only the best OCR configuration
            ocr_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
            
            try:
                text = pytesseract.image_to_string(
                    thresh,
                    config=ocr_config,
                    lang='eng'
                )
                
                # Clean the text and extract number
                text = text.strip()
                
                # Try to extract a number from the text
                import re
                match = re.search(r'\d+', text)
                
                detected_round = None
                if match:
                    round_num = int(match.group())
                    
                    # Basic validation
                    if 1 <= round_num <= 200:
                        detected_round = round_num
                        logging.info(f"Detected round: {detected_round} (raw text: '{text}')")
                    else:
                        logging.warning(f"Invalid round number detected: {round_num} (raw text: '{text}')")
                else:
                    logging.warning(f"No number found in text: '{text}'")
                
                # Save debug screenshot with detected round number (for dataset collection)
                timestamp = int(time.time())
                if detected_round is not None:
                    debug_filename = f'round_{detected_round}_{timestamp}.png'
                    logging.info(f"Round screenshot saved: {debug_filename}")
                else:
                    debug_filename = f'round_none_{timestamp}.png'
                    logging.warning(f"Failed to detect round - screenshot saved: {debug_filename}")
                
                debug_path = f'images/screenshots/{debug_filename}'
                screenshot.save(debug_path)
                
                # Also save the processed image for dataset analysis
                if detected_round is not None:
                    processed_filename = f'round_{detected_round}_{timestamp}_processed.png'
                    processed_path = f'images/screenshots/{processed_filename}'
                    cv2.imwrite(processed_path, thresh)
                
                return detected_round
                
            except Exception as e:
                logging.error(f"OCR failed: {e}")
                
                # Save screenshot even if OCR fails (for dataset analysis)
                timestamp = int(time.time())
                debug_filename = f'round_ocr_error_{timestamp}.png'
                debug_path = f'images/screenshots/{debug_filename}'
                screenshot.save(debug_path)
                
                return None
                
        except Exception as e:
            logging.error(f"Error reading current round: {e}")
            return None

    def execute_round_actions(self, map_name, current_round):
        """Execute tower placements and actions for the current round"""
        try:
            with open('placements.json', 'r') as f:
                all_placements = json.load(f)
            
            # Check if map exists
            if map_name not in all_placements:
                logging.debug(f"Map '{map_name}' not found in placements.json")
                return False
            
            map_data = all_placements[map_name]
            
            # Check if game difficulty exists for this map
            game_difficulty = GAME_DIFFICULTY.lower()
            if game_difficulty not in map_data:
                logging.debug(f"Game difficulty '{game_difficulty}' not found for map '{map_name}'")
                return False
            
            game_difficulty_data = map_data[game_difficulty]
            
            # Check if mode exists for this game difficulty
            mode = MODE.lower()
            if mode not in game_difficulty_data:
                logging.debug(f"Mode '{mode}' not found for map '{map_name}' in '{game_difficulty}' difficulty")
                return False
            
            mode_data = game_difficulty_data[mode]
            rounds_data = mode_data.get("rounds", {})
            
            # Check if there are actions for the current round
            if str(current_round) not in rounds_data:
                logging.debug(f"No tower placements for round {current_round} on map '{map_name}' in '{game_difficulty}' difficulty with '{mode}' mode")
                return False  # No actions for this round
            
            round_actions = rounds_data[str(current_round)]
            gameplay = Gameplay()
            
            logging.info(f"Executing round {current_round} actions for map '{map_name}' in '{game_difficulty}' difficulty with '{mode}' mode")
            
            for action in round_actions:
                try:
                    method = action.get("method", "key")
                    coordinates = tuple(action["coordinates"])
                    action_type = action.get("action", "place")
                    
                    if action_type == "place":
                        if method == "key":
                            key = action["key"]
                            gameplay.place_tower(key, coordinates)
                            logging.info(f"Placed tower with key '{key}' at {coordinates}")
                        elif method == "image":
                            if "image" not in action:
                                logging.error(f"Missing 'image' field in action: {action}")
                                continue
                            image = action["image"]
                            gameplay.place_tower_by_image(image, coordinates)
                            logging.info(f"Placed tower with image '{image}' at {coordinates}")
                        else:
                            logging.warning(f"Unknown placement method: {method} in action: {action}")
                    
                    elif action_type == "upgrade":
                        if "upgrade_path" in action:
                            # Use the new upgrade_tower method with upgrade path
                            upgrade_path = action["upgrade_path"]
                            gameplay.upgrade_tower(coordinates, upgrade_path)
                            logging.info(f"Upgraded tower at {coordinates} with path {upgrade_path}")
                        elif method == "key":
                            key = action["key"]
                            # Click on the tower at coordinates, then press the upgrade key
                            pyautogui.click(coordinates)
                            time.sleep(0.1)
                            keyboard.press_and_release(key)
                            logging.info(f"Upgraded tower at {coordinates} with key '{key}'")
                        else:
                            logging.warning(f"Unknown upgrade method: {method} in action: {action}")
                    
                    elif action_type == "sell":
                        # Click on the tower at coordinates, then press the sell key
                        pyautogui.click(coordinates)
                        time.sleep(0.1)
                        keyboard.press_and_release('backspace')  # Default sell key
                        logging.info(f"Sold tower at {coordinates}")
                    
                    elif action_type == "target":
                        # Click on the tower at coordinates, then press the target key
                        pyautogui.click(coordinates)
                        time.sleep(0.1)
                        target_key = action.get("key", "tab")  # Default target key
                        keyboard.press_and_release(target_key)
                        logging.info(f"Changed target priority at {coordinates} with key '{target_key}'")
                    
                    else:
                        logging.warning(f"Unknown action type: {action_type} in action: {action}")
                    
                    time.sleep(0.5)  # Small delay between actions
                    
                except Exception as e:
                    logging.error(f"Error processing action {action}: {e}")
            
            logging.info(f"Completed round {current_round} actions for map '{map_name}' in '{game_difficulty}' difficulty with '{mode}' mode")
            return True
            
        except Exception as e:
            logging.error(f"Error loading or processing placements.json for round {current_round}: {e}")
            return False

    def monitor_rounds_and_execute(self, map_name):
        """Monitor the current round and execute tower placements when needed - optimized for dataset collection"""
        logging.info(f"Starting round monitoring for map: {map_name}")
        
        last_round = None
        executed_rounds = set()  # Track which rounds we've already executed
        
        # Get all available rounds for this map
        try:
            with open('placements.json', 'r') as f:
                all_placements = json.load(f)
            
            if map_name not in all_placements:
                logging.error(f"Map '{map_name}' not found in placements.json")
                return
            
            map_data = all_placements[map_name]
            game_difficulty = GAME_DIFFICULTY.lower()
            mode = MODE.lower()
            
            if game_difficulty not in map_data or mode not in map_data[game_difficulty]:
                logging.error(f"Configuration not found for map '{map_name}' in '{game_difficulty}' difficulty with '{mode}' mode")
                return
            
            rounds_data = map_data[game_difficulty][mode].get("rounds", {})
            available_rounds = [int(round_num) for round_num in rounds_data.keys()]
            available_rounds.sort()  # Sort rounds in ascending order
            
            # Determine starting round based on difficulty and mode
            if game_difficulty == "hard" and mode == "impoppable":
                starting_round = 6  # Impoppable starts with round 6
                logging.info(f"Impoppable mode detected - starting with round {starting_round}")
            else:
                starting_round = 1  # Easy Standard and others start with round 1
                logging.info(f"Standard mode detected - starting with round {starting_round}")
            
            # Filter rounds to only include those >= starting_round
            available_rounds = [round_num for round_num in available_rounds if round_num >= starting_round]
            
            logging.info(f"Available rounds for execution (starting from {starting_round}): {available_rounds}")
            
        except Exception as e:
            logging.error(f"Error loading placements.json: {e}")
            return
        
        while self.running:
            try:
                # Read current round (optimized for speed)
                current_round = self.read_current_round()
                
                if current_round is None:
                    # If we can't read the round, wait a bit and try again
                    time.sleep(0.1)  # Reduced delay for faster dataset collection
                    continue
                
                # Check if round has changed
                if current_round != last_round:
                    logging.info(f"Round changed from {last_round} to {current_round}")
                    last_round = current_round
                
                # Check which rounds should be executed now
                rounds_to_execute = []
                for target_round in available_rounds:
                    # Only execute if:
                    # 1. Target round is less than or equal to current round
                    # 2. Target round hasn't been executed yet
                    # 3. Current round is at least the starting round
                    if (target_round <= current_round and 
                        target_round not in executed_rounds and 
                        current_round >= starting_round):
                        rounds_to_execute.append(target_round)
                
                # Execute actions for rounds that should be executed now
                for round_to_execute in rounds_to_execute:
                    logging.info(f"Executing actions for round {round_to_execute} (current round: {current_round})")
                    if self.execute_round_actions(map_name, round_to_execute):
                        executed_rounds.add(round_to_execute)
                        logging.info(f"Successfully executed actions for round {round_to_execute}")
                    else:
                        logging.debug(f"No actions to execute for round {round_to_execute}")
                
                # Check if game is finished (look for victory or defeat screen)
                if self.is_game_finished():
                    logging.info("Game finished detected")
                    break
                
                # Reduced delay for faster dataset collection
                time.sleep(0.2)  # Reduced from 0.5 to 0.2 seconds
                
            except KeyboardInterrupt:
                logging.info("Round monitoring interrupted by user")
                break
            except Exception as e:
                logging.error(f"Error in round monitoring: {e}")
                time.sleep(0.5)  # Keep this delay for error recovery
        
        logging.info(f"Round monitoring completed for map: {map_name}")
    
    def is_game_finished(self):
        """Check if the game is finished (victory or defeat)"""
        try:
            # Look for victory or defeat indicators
            # You can add more specific checks here based on your game's UI
            victory_indicators = [
                'images/end/victory.png',
                'images/end/next.png',
                'images/end/restart_button.png',
                'images/misc/level_up.png'
            ]
            
            for indicator in victory_indicators:
                if self.find_image_on_screen(indicator, confidence=0.7, image_name=indicator.split('/')[-1])[0] is not None:
                    return True
            
            return False
            
        except Exception as e:
            logging.error(f"Error checking if game is finished: {e}")
            return False

    def find_map_to_play(self):
        """Find the map to play"""
        self.wait_and_click(f'images/difficulty/{MAP_DIFFICULTY}.png', 'Difficulty button')
        while self.find_image_on_screen(f'images/maps/{MAP_DIFFICULTY}/{DEFAULT_MAP}.png', confidence=0.5)[0] is None :
            time.sleep(0.3)
            self.wait_and_click(f'images/difficulty/{MAP_DIFFICULTY}.png', 'Difficulty button')
            
        self.wait_and_click(f'images/maps/{MAP_DIFFICULTY}/{DEFAULT_MAP}.png', 'Map button', confidence=0.5)

        # Click on the difficulty
        self.wait_and_click(f'images/difficulty/{GAME_DIFFICULTY}.png', f'{GAME_DIFFICULTY} difficulty')
        # Click on the mode
        self.wait_and_click(f'images/difficulty/{MODE}.png', f'Mode: {MODE}')
        # If Impoppable, click on OK to pass the warning
        if MODE in ["impoppable", "deflation"]:
            self.wait_and_click(f'images/misc/ok.png', 'OK button')
        return True

    def finish_game(self, write_in_terminal = False): 
        if self.find_image_on_screen(f'images/end/next.png', confidence=0.7, image_name="next")[0] is not None:
            self.wait_and_click(f'images/end/next.png', 'Finished, click next')
            
            self.game_won = True
            return True
        elif self.find_image_on_screen(f'images/end/restart_button.png', confidence=0.7, image_name="restart")[0] is not None:
            self.wait_and_click(f'images/end/restart_button.png', 'Restart button')
            self.wait_and_click(f'images/end/confirm_restart.png', 'Confirm restart')
            self.game_won = False
            return True
        elif self.find_image_on_screen(f'images/misc/level_up.png', confidence=0.7, image_name="level_up")[0] is not None:
            
            self.wait_and_click(f'images/misc/level_up.png', 'Level up')
            if self.finish_game():
                self.game_won = True
                return True
            else:
                self.game_won = False
                return False
        else:
            self.game_won = False
            return False

    def run_collect_event(self):
        """Main bot loop"""
        print("🎯 Starting bot cycle...")
        self.game_won = False
        # Step 1 : Click Play
        print("▶️ Clicking Play button...")
        self.wait_and_click('images/misc/play_button.png', 'Play button', confidence=0.7) 
        # Step 2 : Click search 
        self.wait_and_click('images/misc/search.png', 'Click search button')
        # Step 2.1 : Click Search collect event
        self.wait_and_click('images/misc/search_collect_event.png', 'Click search collect event')
        # Step 2 : Read the map to play
        # map_name = self.read_which_map_to_play()
        # Step 2.5 : Take screenshot of the map
        self.wait_for_any_image(['images/misc/collection_event_searchbar.png'])
        print("🗺️ Identifying map to play...")
        map_name = self.take_screenshot_and_compare(save_screenshot=False)
        print(f"🎮 Playing map: {map_name}")

        self.navigate_to_map(map_name)
        
        while not self.game_won:
            # Step 3 : Play the map
            self.play_map(map_name)
            # Step 4 : Finish the game
            while not self.finish_game():
                time.sleep(0.5)

        # Step 5 : Click Home 
        print("🏠 Returning to home...")
        self.wait_and_click('images/end/home.png', 'Click Home')
        # self.wait_for_any_image(['images/collection_event/collect.png', 'images/misc/collect_event.png'], 0.8)
        # Step 6 : Check Collect Event
        # print("📦 Checking for collection event...")
        # self.collection_event()
        print("✅ Bot cycle completed!")

    def run_normal_game(self):
        """Main bot loop"""
        print("🎯 Starting bot cycle...")
        self.game_won = False
        # Step 1 : Click Play
        print("▶️ Clicking Play button...")
        self.wait_and_click('images/misc/play_button.png', 'Play button')
        # Step 2 : Click the map difficulty
        self.find_map_to_play()
        while not self.game_won:
            # Step 3 : Play the map
            self.play_map(DEFAULT_MAP)
            # Step 4 : Finish the game
            while not self.finish_game():
                time.sleep(0.5)

        # Step 5 : Click Home 
        print("🏠 Returning to home...")
        self.wait_and_click('images/end/home.png', 'Click Home')
        print("✅ Bot cycle completed!")

if __name__ == "__main__":
    print("=" * 60)
    print("🎮 BTD6 Automation Bot Starting Up")
    print("=" * 60)
    
    # Show current configuration
    print(f"📋 Configuration:")
    print(f"   Difficulty: {GAME_DIFFICULTY}")
    print(f"   Mode: {MODE}")
    print(f"   Config File: {CONFIG_FILE if CONFIG_FILE else 'config.py (default)'}")
    print()
    
    # Show usage if no arguments provided
    if len(sys.argv) == 1:
        print("💡 Usage examples:")
        print("   python main.py                    # Use default config")
        print("   python main.py config_easy        # Use easy difficulty")
        print("   python main.py config_hard        # Use hard difficulty")
        print("   python main.py config_impoppable  # Use impoppable mode")
        print()
    
    bot = BTD6Bot()
    
    print("\n🎮 Launching BTD6...")
    if not bot.launch_game():
        print("❌ Failed to launch game. Exiting...")
        logging.error("Failed to launch game. Exiting...")
        exit(1)
    
    print("\n🎯 Bot is ready! Starting main loop...")
    print("=" * 60)

    # bot.restart_game()
    
    try:
        while True: 
            bot.run_normal_game()
    except KeyboardInterrupt:
        print("\n⏹️ Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Bot crashed: {e}")
        logging.error(f"Bot crashed: {e}")
    finally:
        print("👋 Bot shutdown complete") 