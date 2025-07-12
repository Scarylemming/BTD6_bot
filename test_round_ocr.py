#!/usr/bin/env python3
"""
Test script for round OCR functionality
This script helps debug and improve round number detection
"""

import cv2
import numpy as np
import pytesseract
import os
import logging
from PIL import Image
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_tesseract_installation():
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
        print("3. Or specify the path manually:")
        print("   pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'")
        print("\n🔧 Alternative solutions:")
        print("- Restart your terminal/IDE after installation")
        print("- Add Tesseract installation directory to your system PATH")
        print(f"\n❌ Error details: {e}")
        return False

def preprocess_image_for_ocr(image):
    """
    Apply advanced preprocessing techniques to improve OCR accuracy
    """
    try:
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Scale up the image (sometimes helps with small text)
        scale_factor = 2
        scaled = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(scaled, (3, 3), 0)
        
        # Apply different thresholding methods
        processed_images = []
        
        # 1. Otsu's thresholding
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(("otsu_scaled", otsu))
        
        # 2. Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        processed_images.append(("adaptive_scaled", adaptive))
        
        # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_img = clahe.apply(blurred)
        _, clahe_thresh = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(("clahe", clahe_thresh))
        
        # 4. Morphological operations
        kernel = np.ones((2,2), np.uint8)
        morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
        processed_images.append(("morph", morph))
        
        # 5. Edge enhancement
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(blurred, -1, kernel_sharpen)
        _, sharp_thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(("sharpened", sharp_thresh))
        
        return processed_images
        
    except Exception as e:
        logging.error(f"Error in preprocess_image_for_ocr: {e}")
        return []

def test_round_ocr(image_path):
    """
    Test round OCR on a specific image
    """
    try:
        # Load the image
        screenshot = Image.open(image_path)
        screenshot_np = np.array(screenshot)
        gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)
        
        print(f"Testing OCR on: {image_path}")
        print(f"Image shape: {gray.shape}")
        
        # Try multiple preprocessing techniques
        processed_images = []
        
        # Basic preprocessing techniques
        # 1. Original thresholding
        _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(("otsu", thresh1))
        
        # 2. Adaptive thresholding
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        processed_images.append(("adaptive", thresh2))
        
        # 3. Simple thresholding with different values
        _, thresh3 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        processed_images.append(("simple_127", thresh3))
        
        # 4. Inverted thresholding
        _, thresh4 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        processed_images.append(("otsu_inv", thresh4))
        
        # 5. Morphological operations
        kernel = np.ones((2,2), np.uint8)
        thresh5 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        processed_images.append(("morph_clean", thresh5))
        
        # Add advanced preprocessing techniques
        advanced_processed = preprocess_image_for_ocr(gray)
        processed_images.extend(advanced_processed)
        
        # Try different OCR configurations
        ocr_configs = [
            ("standard", r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'),
            ("single_line", r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'),
            ("single_word", r'--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789'),
            ("raw_line", r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'),
            ("sparse_text", r'--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789'),
            ("uniform_block", r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789')
        ]
        
        all_results = []
        
        # Try all combinations
        for img_name, processed_img in processed_images:
            for config_name, ocr_config in ocr_configs:
                try:
                    text = pytesseract.image_to_string(
                        processed_img,
                        config=ocr_config,
                        lang='eng'
                    )
                    
                    text = text.strip()
                    
                    # Try to extract a number
                    import re
                    match = re.search(r'\d+', text)
                    
                    if match:
                        round_num = int(match.group())
                        confidence = len(match.group()) / max(len(text), 1)
                        
                        all_results.append({
                            'method': f"{img_name}_{config_name}",
                            'text': text,
                            'round': round_num,
                            'confidence': confidence
                        })
                        
                        print(f"  {img_name}_{config_name}: '{text}' -> round {round_num} (confidence: {confidence:.2f})")
                    
                except Exception as e:
                    print(f"  {img_name}_{config_name}: OCR failed - {e}")
        
        # Sort by confidence and show top results
        all_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"\nTop 5 results:")
        for i, result in enumerate(all_results[:5]):
            print(f"  {i+1}. {result['method']}: '{result['text']}' -> round {result['round']} (confidence: {result['confidence']:.2f})")
        
        # Save processed images for visual inspection
        output_dir = "ocr_debug_output"
        os.makedirs(output_dir, exist_ok=True)
        
        for img_name, processed_img in processed_images:
            output_path = os.path.join(output_dir, f"{os.path.basename(image_path)}_{img_name}.png")
            cv2.imwrite(output_path, processed_img)
            print(f"Saved processed image: {output_path}")
        
        return all_results
        
    except Exception as e:
        print(f"Error testing OCR: {e}")
        return []

def main():
    """
    Main function to test round OCR
    """
    print("Round OCR Test Script")
    print("=" * 50)
    
    # Check Tesseract installation first
    if not check_tesseract_installation():
        print("\n❌ Cannot run OCR tests without Tesseract. Please install Tesseract first.")
        return
    
    # Check if images/screenshots directory exists
    screenshots_dir = "images/screenshots"
    if not os.path.exists(screenshots_dir):
        print(f"Directory {screenshots_dir} not found!")
        return
    
    # Find round screenshot files
    round_files = [f for f in os.listdir(screenshots_dir) if f.startswith("debug_round_screenshot_")]
    
    if not round_files:
        print("No round screenshot files found!")
        print("Run the main bot first to generate some round screenshots.")
        return
    
    print(f"Found {len(round_files)} round screenshot files:")
    for i, file in enumerate(round_files[:10]):  # Show first 10
        print(f"  {i+1}. {file}")
    
    if len(round_files) > 10:
        print(f"  ... and {len(round_files) - 10} more")
    
    # Test the most recent file
    latest_file = sorted(round_files)[-1]
    latest_path = os.path.join(screenshots_dir, latest_file)
    
    print(f"\nTesting latest file: {latest_file}")
    results = test_round_ocr(latest_path)
    
    if results:
        best_result = results[0]
        print(f"\nBest result: {best_result['method']}")
        print(f"Detected round: {best_result['round']}")
        print(f"Raw text: '{best_result['text']}'")
        print(f"Confidence: {best_result['confidence']:.2f}")
    
    print(f"\nProcessed images saved to: ocr_debug_output/")
    print("You can visually inspect these to see which preprocessing works best.")

if __name__ == "__main__":
    main() 