# BTD6 Bot Configuration System

## Overview
The BTD6 bot now supports multiple configuration files that can be specified via command line arguments. This allows you to easily switch between different game difficulties and modes without modifying the main code.

## Available Config Files

### 1. `config.py` (Default)
- **Difficulty**: Impoppable
- **Mode**: Standard
- **Use**: `python main.py` or `python main.py config`

### 2. `config_easy.py`
- **Difficulty**: Easy
- **Mode**: Standard
- **Use**: `python main.py config_easy`

### 3. `config_hard.py`
- **Difficulty**: Hard
- **Mode**: Standard
- **Use**: `python main.py config_hard`

### 4. `config_impoppable.py`
- **Difficulty**: Hard
- **Mode**: Impoppable
- **Use**: `python main.py config_impoppable`

## Usage Examples

```bash
# Use default config (Impoppable + Standard)
python main.py

# Use easy difficulty
python main.py config_easy

# Use hard difficulty
python main.py config_hard

# Use impoppable mode
python main.py config_impoppable

# Use custom config file
python main.py my_custom_config
```

## Creating Custom Config Files

You can create your own config files by copying one of the existing ones and modifying the settings:

1. **Copy an existing config file**:
   ```bash
   cp config_easy.py my_config.py
   ```

2. **Edit the settings** in `my_config.py`:
   ```python
   GAME_DIFFICULTY = "Hard"   # Change difficulty
   MODE = "CHIMPS"           # Change mode
   ROUNDS_TO_PLAY = 30       # Change rounds
   ```

3. **Use your custom config**:
   ```bash
   python main.py my_config
   ```

## Configuration Options

Each config file contains these main settings:

### Game Settings
- `GAME_DIFFICULTY`: "Easy", "Hard", etc. (affects tower placement strategy)
- `MODE`: "Standard", "Impoppable", "CHIMPS", etc. (affects game rules)
- `ROUNDS_TO_PLAY`: Number of rounds to play before restarting

### Bot Settings
- `WAIT_TIME`: Delay between actions (seconds)
- `CONFIDENCE_THRESHOLD`: Image recognition confidence (0.0-1.0)
- `GAME_PATH`: Path to BTD6 executable

### Safety Settings
- `EMERGENCY_STOP_REGION`: Screen region for emergency stop
- `RUN_IN_BACKGROUND`: Whether to run bot in background

## JSON Structure Compatibility

The config system works with the new JSON structure:

```json
{
    "map_name": {
        "easy": {                    // GAME_DIFFICULTY
            "standard": {            // MODE
                "rounds": {
                    "1": [
                        {
                            "method": "image",
                            "image": "images/towers/u.png",
                            "coordinates": [x, y],
                            "action": "place"
                        }
                    ]
                }
            }
        }
    }
}
```

## Error Handling

- If a config file is not found, the bot will fall back to the default `config.py`
- If no config file is specified, the bot uses `config.py` by default
- The bot will show which config file is being used at startup

## Tips

1. **Test different strategies**: Use different config files to test various tower placement strategies
2. **Quick switching**: Easily switch between easy and hard difficulties for testing
3. **Custom setups**: Create config files for specific maps or strategies
4. **Backup configs**: Keep backup config files for different game versions or setups

## Example Workflow

```bash
# Start with easy difficulty to test
python main.py config_easy

# Switch to hard difficulty
python main.py config_hard

# Try impoppable mode
python main.py config_impoppable

# Use custom strategy
python main.py my_strategy_config
``` 