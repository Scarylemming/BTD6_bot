from main import BTD6Bot
import time
import logging

# Set up logging to see debug information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

bot = BTD6Bot()
time.sleep(3)

print("Testing debug template matching...")
matches = bot.debug_template_matching("tier2", threshold=0.3)

if matches:
    print(f"Found {len(matches)} matches:")
    for i, (confidence, (x, y), (w, h)) in enumerate(matches):
        center_x = x + w // 2
        center_y = y + h // 2
        print(f"Match {i+1}: confidence={confidence:.3f}, coords=({x}, {y}), center=({center_x}, {center_y})")
else:
    print("No matches found")

print("\nTesting collection event with debugging...")
bot.collection_event() 