# Round OCR Improvements

## Overview

The round reading function has been significantly improved to provide better accuracy and debugging capabilities. The main improvements include:

1. **Multiple preprocessing techniques** - The system now tries 10+ different image preprocessing methods
2. **Multiple OCR configurations** - 6 different Tesseract OCR configurations are tested
3. **Debug screenshots with round numbers** - Screenshots are saved with the detected round number in the filename
4. **Validation and confidence scoring** - Round numbers are validated and ranked by confidence
5. **Comprehensive logging** - All OCR attempts are logged for debugging

## New Features

### 1. Debug Screenshots with Round Numbers

Screenshots are now saved with the detected round number in the filename:
- `debug_round_screenshot_1234567890_Round_6.png` - Successfully detected round 6
- `debug_round_screenshot_1234567890_Round_None.png` - Failed to detect any round

### 2. Processed Image Debugging

The best processed image is also saved for visual inspection:
- `debug_round_processed_1234567890_Round_6_otsu.png` - Shows the preprocessing that worked best

### 3. Comprehensive Logging

The system now logs:
- All OCR attempts with their results
- Top 5 best results with confidence scores
- Which preprocessing method worked best
- Validation warnings for unusual round numbers

## How to Help Improve OCR Accuracy

### 1. Use the Test Script

Run the test script to analyze existing screenshots:

```bash
python test_round_ocr.py
```

This will:
- Find all round screenshots in `images/screenshots/`
- Test all preprocessing and OCR combinations
- Save processed images to `ocr_debug_output/`
- Show which methods work best

### 2. Visual Inspection

After running the test script, check the `ocr_debug_output/` directory to see:
- How different preprocessing techniques affect the image
- Which preprocessing makes the numbers most readable
- Which methods consistently produce good results

### 3. Adjust Screenshot Region

If OCR is still unreliable, you can adjust the screenshot region in `main.py`:

```python
# Current region: (1365, 80, 76, 42)
screenshot = pyautogui.screenshot(region=(1365, 80, 76, 42))
```

Try different regions to capture the round number more clearly.

### 4. Add Custom Preprocessing

You can add new preprocessing techniques in the `preprocess_image_for_ocr()` function:

```python
# Example: Add custom thresholding
_, custom_thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
processed_images.append(("custom_100", custom_thresh))
```

### 5. Adjust OCR Configurations

Modify the OCR configurations in `read_current_round()`:

```python
ocr_configs = [
    # Add your custom configurations here
    ("custom", r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789 --tessdata-dir /path/to/custom/tessdata'),
]
```

## Preprocessing Techniques Used

### Basic Techniques
1. **Otsu Thresholding** - Automatic threshold detection
2. **Adaptive Thresholding** - Local threshold adaptation
3. **Simple Thresholding** - Fixed threshold at 127
4. **Inverted Thresholding** - For dark text on light backgrounds
5. **Morphological Operations** - Noise reduction and cleanup

### Advanced Techniques
1. **Image Scaling** - 2x upscaling for better OCR
2. **Gaussian Blur** - Noise reduction
3. **CLAHE** - Contrast enhancement
4. **Edge Enhancement** - Sharpening filter
5. **Morphological Closing** - Fill gaps in characters

## OCR Configurations

1. **Standard** - `--psm 7` - Single text line
2. **Single Line** - `--psm 8` - Single word
3. **Single Word** - `--psm 13` - Raw line
4. **Raw Line** - `--psm 6` - Uniform block
5. **Sparse Text** - `--psm 11` - Sparse text
6. **Uniform Block** - `--psm 6` - Single uniform block

## Validation Features

The system now validates round numbers:
- Must be positive integers
- Must be between 1 and 200
- Warns about unusual decreases in round numbers
- Tracks the last valid round for consistency checking

## Troubleshooting

### If OCR is still unreliable:

1. **Check the debug screenshots** - Look at the saved images to see if the round number is clearly visible
2. **Adjust the screenshot region** - The round number might be in a different location
3. **Check game resolution** - Different resolutions might require different regions
4. **Try different preprocessing** - Use the test script to find which preprocessing works best
5. **Check Tesseract installation** - Ensure Tesseract is properly installed and configured

### Common Issues:

1. **Round number too small** - Try increasing the screenshot region or using scaling
2. **Poor contrast** - Try different thresholding methods
3. **Noise in image** - Try morphological operations or blurring
4. **Wrong font** - Tesseract might not recognize the game's font well

## Performance Notes

- The improved OCR is slower due to multiple attempts, but more accurate
- Debug images are only saved when a round is detected or when debugging is enabled
- The system caches the last valid round for validation
- All preprocessing attempts are logged for analysis

## Future Improvements

Potential areas for further improvement:
1. **Machine learning** - Train a custom OCR model on BTD6 round numbers
2. **Template matching** - Create templates for each digit (0-9)
3. **Color-based detection** - Use color information to improve accuracy
4. **Dynamic region detection** - Automatically find the round number location
5. **Confidence thresholds** - Adjust thresholds based on game difficulty/mode 