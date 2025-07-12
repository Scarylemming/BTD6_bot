# BTD6 Bot - Tower Upgrade Functionality

This document explains how to use the new tower upgrade functionality in the BTD6 automation bot.

## Overview

The bot now supports automatic tower upgrades during gameplay. You can specify when to upgrade towers, which towers to upgrade, and what upgrade path to follow.

## Upgrade System

### Upgrade Keys
- **Path 1**: `,` (comma key)
- **Path 2**: `.` (period key)  
- **Path 3**: `-` (minus key)

### Upgrade Path Format
The upgrade path is specified as a list of 3 integers: `[path1, path2, path3]`

Each integer represents the number of upgrades to apply to that path.

**Examples:**
- `[1, 2, 0]` = 1 upgrade on path 1, 2 upgrades on path 2, 0 on path 3
- `[0, 0, 2]` = 2 upgrades on path 3 only
- `[2, 1, 0]` = 2 upgrades on path 1, 1 on path 2, 0 on path 3

## JSON Configuration

### Basic Upgrade Action
```json
{
    "action": "upgrade",
    "coordinates": [649, 834],
    "upgrade_path": [1, 2, 0]
}
```

### Complete Example
```json
{
    "end_of_the_road": {
        "hard": {
            "impoppable": {
                "rounds": {
                    "43": [
                        {
                            "action": "upgrade",
                            "coordinates": [649, 834],
                            "upgrade_path": [1, 2, 0]
                        }
                    ],
                    "50": [
                        {
                            "action": "upgrade", 
                            "coordinates": [763, 508],
                            "upgrade_path": [0, 0, 2]
                        }
                    ]
                }
            }
        }
    }
}
```

## How It Works

1. **Tower Selection**: The bot clicks on the tower at the specified coordinates
2. **Upgrade Application**: For each path with upgrades > 0, it presses the corresponding key the specified number of times
3. **Timing**: There's a 0.2-second delay between each upgrade to ensure the game registers the input

## Usage Examples

### Example 1: Basic Tower Upgrade
```json
"43": [
    {
        "action": "upgrade",
        "coordinates": [649, 834],
        "upgrade_path": [1, 2, 0]
    }
]
```
This will upgrade a tower at coordinates (649, 834) on round 43 with:
- 1 upgrade on path 1 (press `,` once)
- 2 upgrades on path 2 (press `.` twice)
- 0 upgrades on path 3 (no action)

### Example 2: Multiple Upgrades in One Round
```json
"50": [
    {
        "action": "upgrade",
        "coordinates": [763, 508],
        "upgrade_path": [0, 0, 2]
    },
    {
        "action": "upgrade",
        "coordinates": [649, 834],
        "upgrade_path": [2, 0, 0]
    }
]
```
This will upgrade two different towers on round 50.

## Testing the Upgrade Functionality

You can test the upgrade functionality using the provided test script:

```bash
python test_upgrade.py
```

This script will:
1. Show you the upgrade paths it will test
2. Ask for confirmation before starting
3. Test different upgrade configurations
4. Provide detailed logging of what's happening

## Important Notes

1. **Coordinates**: Make sure the coordinates point to an actual tower in your game
2. **Timing**: The bot waits 0.3 seconds after clicking a tower before applying upgrades
3. **Game State**: Ensure the game is in a state where upgrades can be applied
4. **Money**: Make sure you have enough money for the upgrades you're requesting

## Troubleshooting

### Common Issues

1. **Tower not found**: Check that the coordinates are correct
2. **Upgrades not applying**: Ensure you have enough money and the tower is selected
3. **Wrong upgrade path**: Verify the upgrade_path array has exactly 3 integers

### Debug Information

The bot logs detailed information about upgrade actions:
- Tower coordinates being clicked
- Upgrade path being applied
- Number of upgrades per path
- Success/failure of each action

Check the `bot.log` file for detailed information about upgrade operations.

## Integration with Existing Bot

The upgrade functionality is fully integrated with the existing bot system:

1. **Round Monitoring**: Upgrades are applied when the specified round is reached
2. **Action Execution**: Upgrades are executed along with other round actions
3. **Error Handling**: Failed upgrades are logged and don't stop the bot
4. **Safety**: The bot continues to function even if upgrades fail

The upgrade system works seamlessly with the existing tower placement and game monitoring systems. 