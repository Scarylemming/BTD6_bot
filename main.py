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
                    time.sleep(5)
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
            screen = pyautogui.screenshot(region=SCREENSHOT_REGION)
            screen_np = np.array(screen)
            template = cv2.imread(image_path)
            if template is None:
                logging.error(f"Template image not found: {image_path}")
                return None, None
            th, tw = template.shape[:2]
            result = cv2.matchTemplate(screen_np, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if max_val >= confidence:
                center_x = max_loc[0] + tw // 2
                center_y = max_loc[1] + th // 2
                return (center_x, center_y), max_val
            print("Max val is ", max_val)
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

    def collection_event(self):
        if self.find_image_on_screen('images/collection_event/collection_event.png')[0] is not None:
            print("Collection Event found")
            self.wait_and_click('images/collection_event/collect.png', 'Click Collection Event')
            for i in range(2):
                self.wait_and_click('images/collection_event/common.png', 'Collect tower')
                time.sleep(0.5)
                pyautogui.click(1000, 1000)
            self.wait_and_click('images/collection_event/continue.png', 'Back to collection event')
            if self.find_image_on_screen('images/collection_event/collection_event.png')[0] is not None and self.find_image_on_screen('images/collection_event/collect.png')[0] is None:
                pyautogui.click(84,67)
            self.wait_and_click('images/collection_event/back.png', 'Back to home')
        return True

    def run(self):
        """Main bot loop"""
        logging.info("Starting BTD6 step-by-step bot.")
        if not self.launch_game():
            logging.error("Failed to launch game. Exiting...")
            return
        # Step 1: Click Play
        self.wait_and_click('images/play_button.png', 'Play button')
        for i in range(4) : 
            if self.wait_and_click(f'images/maps/{DEFAULT_MAP}.png', f'Map: {DEFAULT_MAP}') :
                break
            self.wait_and_click(f'images/map_difficulty/Beginner.png', f'Beginner button')
        self.wait_and_click(f'images/difficulty/{GAME_DIFFICULTY}.png', 
        f'Difficulty: {GAME_DIFFICULTY}')
        self.wait_and_click(f'images/difficulty/{MODE}.png', f'Mode: {MODE}')
        self.wait_and_click(f'images/misc/round.png', 'Find round to start placing towers')
        logging.info("Step-by-step navigation complete. Ready for next steps.")

        # Place towers from placements.json
        try:
            with open('placements.json', 'r') as f:
                placements = json.load(f)
            gameplay = Gameplay()
            for placement in placements:
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
            logging.info("All towers placed.")
        except Exception as e:
            logging.error(f"Error loading or processing placements.json: {e}")
        # Run game
        keyboard.press_and_release('space')
        time.sleep(0.1)
        keyboard.press_and_release('space')
        # gameplay.run_screenshot_loop()
        self.wait_and_click('images/end/next.png', 'Finished, click next')
        self.wait_and_click('images/end/home.png', 'Click Home')
        print("Finished")
        time.sleep(2)
        self.collection_event()
        

if __name__ == "__main__":
    bot = BTD6Bot()
    while True : 
        bot.run() 