# Game window settings
GAME_WINDOW_TITLE = "BloonsTD6"
GAME_WINDOW_SIZE = (1280, 720)  # Default windowed mode size

# Background operation settings
RUN_IN_BACKGROUND = True  # Set to True to run bot behind the scenes
MINIMIZE_AFTER_LAUNCH = True  # Minimize game window after launch
RESTORE_WINDOW_BEFORE_ACTION = True  # Restore window before taking actions
MINIMIZE_AFTER_ACTION = True  # Minimize window after completing actions

# Game installation path
GAME_PATH = r"C:\Program Files (x86)\Steam\steamapps\common\BloonsTD6\BloonsTD6.exe"  # Default Steam installation path

# Map settings
MAP_DIFFICULTY = "beginner"
DEFAULT_MAP = "logs"  # Default map to play
GAME_DIFFICULTY = "easy"   # Game difficulty
MODE = "deflation"     # Game mode

# Tower settings
DEFAULT_TOWERS = {
    "Dart Monkey": (400, 400),  # Position for first tower
    "Ninja Monkey": (500, 400), # Position for second tower
    "Alchemist": (450, 450)     # Position for third tower
}

# Game settings
ROUNDS_TO_PLAY = 20   # Number of rounds to play before restarting
WAIT_TIME = 0.1       # Default wait time between actions (seconds)

# Image recognition settings
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for image recognition
SCREENSHOT_REGION = None    # Region to capture (None = full screen)

# Safety settings
EMERGENCY_STOP_REGION = (0, 0, 100, 100)  # Top-left corner region for emergency stop 

# Tesseract OCR settings
# Uncomment and modify the path below if Tesseract is not in your system PATH
# TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows example
# TESSERACT_PATH = "/usr/bin/tesseract"  # Linux/macOS example 