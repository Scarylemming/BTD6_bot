# -*- coding: utf-8 -*-
"""
Dataset Organization Script for BTD6 Round Detection AI Training

This script organizes round screenshots into a proper dataset structure
for training a machine learning model to detect round numbers.
"""

import os
import shutil
import re
from collections import defaultdict
import json

def organize_round_dataset():
    """
    Organize round screenshots into a proper dataset structure
    """
    print("🔧 Organizing BTD6 Round Dataset")
    print("=" * 50)
    
    # Source directory
    screenshots_dir = "images/screenshots"
    if not os.path.exists(screenshots_dir):
        print(f"❌ Screenshots directory not found: {screenshots_dir}")
        return
    
    # Create dataset directory structure
    dataset_dir = "round_dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create subdirectories
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    test_dir = os.path.join(dataset_dir, "test")
    
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Find all round screenshots
    round_files = []
    for file in os.listdir(screenshots_dir):
        if file.startswith("round_") and file.endswith(".png"):
            round_files.append(file)
    
    print(f"📊 Found {len(round_files)} round screenshots")
    
    if not round_files:
        print("❌ No round screenshots found!")
        print("Run the bot first to generate some round screenshots.")
        return
    
    # Parse round numbers and organize files
    round_data = defaultdict(list)
    failed_detections = []
    
    for file in round_files:
        # Extract round number from filename
        # Expected format: round_X_timestamp.png or round_none_timestamp.png
        match = re.match(r'round_(\d+)_(\d+)\.png', file)
        if match:
            round_num = int(match.group(1))
            timestamp = match.group(2)
            round_data[round_num].append({
                'filename': file,
                'timestamp': timestamp,
                'path': os.path.join(screenshots_dir, file)
            })
        else:
            # Check for failed detections
            if file.startswith("round_none_") or file.startswith("round_ocr_error_"):
                failed_detections.append(file)
    
    print(f"✅ Successfully parsed {len(round_data)} different round numbers")
    print(f"❌ Found {len(failed_detections)} failed detections")
    
    # Show statistics
    print("\n📈 Round Distribution:")
    for round_num in sorted(round_data.keys()):
        count = len(round_data[round_num])
        print(f"  Round {round_num}: {count} images")
    
    # Split data into train/val/test (80/10/10 split)
    total_images = sum(len(images) for images in round_data.values())
    train_count = int(total_images * 0.8)
    val_count = int(total_images * 0.1)
    test_count = total_images - train_count - val_count
    
    print(f"\n📊 Dataset Split:")
    print(f"  Train: {train_count} images")
    print(f"  Validation: {val_count} images")
    print(f"  Test: {test_count} images")
    
    # Create dataset metadata
    dataset_info = {
        "total_images": total_images,
        "unique_rounds": len(round_data),
        "round_distribution": {str(k): len(v) for k, v in round_data.items()},
        "failed_detections": len(failed_detections),
        "split": {
            "train": train_count,
            "validation": val_count,
            "test": test_count
        }
    }
    
    # Organize files into splits
    train_files = []
    val_files = []
    test_files = []
    
    for round_num, images in round_data.items():
        # Sort by timestamp to ensure consistent splits
        images.sort(key=lambda x: x['timestamp'])
        
        # Split images for this round
        total_for_round = len(images)
        train_for_round = int(total_for_round * 0.8)
        val_for_round = int(total_for_round * 0.1)
        
        # Add to train set
        train_files.extend(images[:train_for_round])
        
        # Add to validation set
        val_files.extend(images[train_for_round:train_for_round + val_for_round])
        
        # Add to test set
        test_files.extend(images[train_for_round + val_for_round:])
    
    # Copy files to appropriate directories
    print("\n📁 Copying files to dataset directories...")
    
    # Copy train files
    for i, img_data in enumerate(train_files):
        src = img_data['path']
        dst = os.path.join(train_dir, f"round_{img_data['filename']}")
        shutil.copy2(src, dst)
        if (i + 1) % 10 == 0:
            print(f"  Copied {i + 1}/{len(train_files)} train files")
    
    # Copy validation files
    for i, img_data in enumerate(val_files):
        src = img_data['path']
        dst = os.path.join(val_dir, f"round_{img_data['filename']}")
        shutil.copy2(src, dst)
        if (i + 1) % 10 == 0:
            print(f"  Copied {i + 1}/{len(val_files)} validation files")
    
    # Copy test files
    for i, img_data in enumerate(test_files):
        src = img_data['path']
        dst = os.path.join(test_dir, f"round_{img_data['filename']}")
        shutil.copy2(src, dst)
        if (i + 1) % 10 == 0:
            print(f"  Copied {i + 1}/{len(test_files)} test files")
    
    # Save dataset metadata
    metadata_path = os.path.join(dataset_dir, "dataset_info.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Create labels file for training
    create_labels_file(dataset_dir, train_files, val_files, test_files)
    
    print(f"\n✅ Dataset organized successfully!")
    print(f"📁 Dataset location: {dataset_dir}")
    print(f"📄 Metadata saved to: {metadata_path}")
    print(f"📊 Total images organized: {total_images}")
    
    # Show final statistics
    print(f"\n📈 Final Dataset Statistics:")
    print(f"  Train images: {len(train_files)}")
    print(f"  Validation images: {len(val_files)}")
    print(f"  Test images: {len(test_files)}")
    print(f"  Unique rounds: {len(round_data)}")
    print(f"  Failed detections: {len(failed_detections)}")

def create_labels_file(dataset_dir, train_files, val_files, test_files):
    """
    Create labels file for training
    """
    labels = {}
    
    # Add train labels
    for img_data in train_files:
        filename = f"round_{img_data['filename']}"
        round_num = int(re.match(r'round_(\d+)_', img_data['filename']).group(1))
        labels[filename] = round_num
    
    # Add validation labels
    for img_data in val_files:
        filename = f"round_{img_data['filename']}"
        round_num = int(re.match(r'round_(\d+)_', img_data['filename']).group(1))
        labels[filename] = round_num
    
    # Add test labels
    for img_data in test_files:
        filename = f"round_{img_data['filename']}"
        round_num = int(re.match(r'round_(\d+)_', img_data['filename']).group(1))
        labels[filename] = round_num
    
    # Save labels
    labels_path = os.path.join(dataset_dir, "labels.json")
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2)
    
    print(f"📝 Labels saved to: {labels_path}")

def show_dataset_stats():
    """
    Show statistics about the current dataset
    """
    dataset_dir = "round_dataset"
    if not os.path.exists(dataset_dir):
        print("❌ Dataset directory not found. Run organize_round_dataset() first.")
        return
    
    metadata_path = os.path.join(dataset_dir, "dataset_info.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
        
        print("📊 Dataset Statistics:")
        print("=" * 30)
        print(f"Total images: {dataset_info['total_images']}")
        print(f"Unique rounds: {dataset_info['unique_rounds']}")
        print(f"Failed detections: {dataset_info['failed_detections']}")
        print(f"Train split: {dataset_info['split']['train']}")
        print(f"Validation split: {dataset_info['split']['validation']}")
        print(f"Test split: {dataset_info['split']['test']}")
        
        print("\n📈 Round Distribution:")
        for round_num, count in sorted(dataset_info['round_distribution'].items(), key=lambda x: int(x[0])):
            print(f"  Round {round_num}: {count} images")
    else:
        print("❌ Dataset metadata not found.")

if __name__ == "__main__":
    print("BTD6 Round Dataset Organizer")
    print("=" * 40)
    
    # Check if screenshots exist
    screenshots_dir = "images/screenshots"
    if not os.path.exists(screenshots_dir):
        print(f"❌ Screenshots directory not found: {screenshots_dir}")
        print("Run the bot first to generate round screenshots.")
    else:
        # Show current stats
        print("📊 Current screenshots:")
        round_files = [f for f in os.listdir(screenshots_dir) if f.startswith("round_") and f.endswith(".png")]
        print(f"  Found {len(round_files)} round screenshots")
        
        if round_files:
            # Organize dataset
            organize_round_dataset()
            
            # Show final stats
            print("\n" + "=" * 50)
            show_dataset_stats()
        else:
            print("❌ No round screenshots found!")
            print("Run the bot first to generate some round screenshots.")