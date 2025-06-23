# BTD6 Collection Event Bot

This bot automates the process of collecting lootboxes in Bloons TD 6 by automatically playing and completing maps.

# Conda environment : btd6_bot

## Setup

1. Install Python 3.8 or higher
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
   Or run the installation script:
   ```
   python install_dependencies.py
   ```
3. Configure the settings in `config.py`
4. Make sure BTD6 is running in windowed mode
5. Run the bot:
   ```
   python main.py
   ```

## Features

- **Background Operation**: The bot can now run behind the scenes, allowing you to use your computer while it works
- Automatic map selection
- Automatic tower placement
- Automatic round progression
- Lootbox collection
- Error handling and recovery
- Window management (minimize/restore as needed)

## Background Mode Configuration

The bot can be configured to run in the background in `config.py`:

```python
# Background operation settings
RUN_IN_BACKGROUND = True  # Set to True to run bot behind the scenes
MINIMIZE_AFTER_LAUNCH = True  # Minimize game window after launch
RESTORE_WINDOW_BEFORE_ACTION = True  # Restore window before taking actions
MINIMIZE_AFTER_ACTION = True  # Minimize window after completing actions
```

### Background Mode Benefits:
- **Use your computer normally**: The bot minimizes the game window when not needed
- **Automatic window management**: The bot restores the window only when taking actions
- **Non-intrusive operation**: You can work, browse, or play other games while the bot runs
- **Efficient resource usage**: The game window is only active when necessary

## Safety Notes

- The bot uses image recognition to interact with the game
- Make sure to run the game in windowed mode
- The bot can be stopped at any time by moving your mouse to the top-left corner of the screen
- In background mode, the game window will be automatically managed

## Disclaimer

This bot is for educational purposes only. Use at your own risk and in accordance with the game's terms of service. 