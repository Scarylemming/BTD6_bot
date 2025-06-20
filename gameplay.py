import os
import time
import pyautogui
import pytesseract
from PIL import Image
import logging
import keyboard
import cv2
import numpy as np
import re

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class Gameplay:
    def __init__(self, screenshot_dir='images/screenshots', interval=5, round_region=(1379, 78, 171, 50)):
        self.screenshot_dir = screenshot_dir
        self.interval = interval  # seconds
        self.round_region = round_region
        os.makedirs(self.screenshot_dir, exist_ok=True)

    def take_screenshot(self, tag=None, region=None):
        timestamp = int(time.time())
        filename = f"screenshot_{tag or timestamp}.png"
        path = os.path.join(self.screenshot_dir, filename)
        region_to_use = region if region is not None else self.round_region
        screenshot = pyautogui.screenshot(region=region_to_use)
        screenshot.save(path)
        logging.info(f"Screenshot saved: {path}")
        return path

    def extract_round_number(self, image_path, region=None):
        img = Image.open(image_path)
        # No need to crop, as screenshot is already cropped
        text = pytesseract.image_to_string(img, config='--psm 7')
        logging.info(f"OCR result: {text}")
        match = re.search(r'\d+', text)
        
        if match:
            round_num = int(match.group())
            logging.info(f"Detected round: {round_num}")
            return round_num
        logging.warning("Could not detect round number.")
        return None

    def run_screenshot_loop(self, duration=60):
        start = time.time()
        while time.time() - start < duration:
            path = self.take_screenshot(region=self.round_region)
            self.extract_round_number(path)
            time.sleep(self.interval)

    def place_tower(self, key, coordinates):
        """
        Place a tower by pressing the key (using keyboard library), moving to coordinates, clicking, and clicking again to select for upgrade.
        Args:
            key (str): The key to press to select the tower (e.g., 'U', 'A', 'S').
            coordinates (tuple): (x, y) screen coordinates to place the tower.
        """
        logging.info(f"Placing tower with key '{key}' at {coordinates}")
        time.sleep(0.2)
        pyautogui.moveTo(coordinates[0], coordinates[1], duration=0.2)
        keyboard.press_and_release(key)
        pyautogui.click()
        time.sleep(0.2)
        pyautogui.click()  # Click again to select the placed tower
        logging.info(f"Tower placed and selected at {coordinates}")

    def place_tower_by_image(self, image_path, coordinates, confidence=0.7):
        """
        Place a tower by finding an image on screen, clicking it, moving to coordinates, and clicking to place the tower.
        Args:
            image_path (str): Path to the image to find and click (e.g., 'u.png').
            coordinates (tuple): (x, y) screen coordinates to place the tower.
            confidence (float): Confidence threshold for image matching.
        """
        logging.info(f"Placing tower by image '{image_path}' at {coordinates}")
        # Take a screenshot
        screen = pyautogui.screenshot()
        screen_np = np.array(screen)
        template = cv2.imread(image_path)
        if template is None:
            logging.error(f"Template image not found: {image_path}")
            return False
        th, tw = template.shape[:2]
        result = cv2.matchTemplate(screen_np, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val >= confidence:
            center_x = max_loc[0] + tw // 2
            center_y = max_loc[1] + th // 2
            pyautogui.moveTo(center_x, center_y, duration=0.2)
            pyautogui.click()
            time.sleep(0.1)
            pyautogui.moveTo(coordinates[0], coordinates[1], duration=0.2)
            pyautogui.click()
            logging.info(f"Tower placed and selected at {coordinates}")
            return True
        else:
            logging.error(f"Could not find image {image_path} on screen (max_val={max_val})")
            return False 

    