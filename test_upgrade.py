#!/usr/bin/env python3
"""
Test script to demonstrate the tower upgrade functionality.
This script shows how to use the upgrade_tower method from the Gameplay class.
"""

import time
import logging
from gameplay import Gameplay

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_upgrade_functionality():
    """Test the upgrade functionality with different upgrade paths"""
    
    print("🧪 Testing Tower Upgrade Functionality")
    print("=" * 50)
    
    # Initialize gameplay
    gameplay = Gameplay()
    
    # Test coordinates (you can change these to match your game)
    test_coordinates = (500, 500)
    
    # Test different upgrade paths
    upgrade_paths = [
        [1, 2, 0],  # 1 upgrade on path 1, 2 on path 2, 0 on path 3
        [0, 0, 2],  # 2 upgrades on path 3 only
        [2, 1, 0],  # 2 on path 1, 1 on path 2
        [1, 1, 1],  # 1 upgrade on each path
    ]
    
    print("📋 Upgrade paths to test:")
    for i, path in enumerate(upgrade_paths, 1):
        print(f"   {i}. Path {path} (Path1: {path[0]}, Path2: {path[1]}, Path3: {path[2]})")
    
    print(f"\n🎯 Test coordinates: {test_coordinates}")
    print("\n⚠️  WARNING: This will attempt to click and upgrade towers!")
    print("   Make sure BTD6 is running and you have a tower at the test coordinates.")
    
    input("\nPress Enter to start testing (or Ctrl+C to cancel)...")
    
    try:
        for i, upgrade_path in enumerate(upgrade_paths, 1):
            print(f"\n🔄 Test {i}: Upgrading with path {upgrade_path}")
            print(f"   This will apply {upgrade_path[0]} upgrade(s) to path 1 (, key)")
            print(f"   This will apply {upgrade_path[1]} upgrade(s) to path 2 (. key)")
            print(f"   This will apply {upgrade_path[2]} upgrade(s) to path 3 (- key)")
            
            # Wait a moment before starting
            time.sleep(2)
            
            # Perform the upgrade
            gameplay.upgrade_tower(test_coordinates, upgrade_path)
            
            print(f"✅ Test {i} completed!")
            
            # Wait between tests
            time.sleep(3)
            
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
    
    print("\n🎉 Upgrade functionality testing completed!")

if __name__ == "__main__":
    test_upgrade_functionality() 