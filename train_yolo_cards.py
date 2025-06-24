#!/usr/bin/env python3
"""
Train YOLO model for playing card detection
Based on AlexeyAB/darknet and your training parameters
"""

import os
import subprocess
import sys

# Configuration from your training image
CONFIG = {
    "darknet_path": "./darknet/darknet",
    "cfg_file": "cfg/yolov3-cards.cfg",
    "data_file": "data/cards.data",
    "pretrained_weights": "darknet53.conv.74",
    "backup_dir": "backup/"
}

# Training parameters from your image
PARAMS = {
    "batch": 64,
    "subdivisions": 16,
    "width": 416,
    "height": 416,
    "flip": 0,  # No flipping for cards!
    "learning_rate": 0.001,
    "classes": 52,
    "filters": 171  # (52+5)*3
}

def setup_directories():
    """Create necessary directories"""
    os.makedirs("cfg", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("backup", exist_ok=True)

def create_data_file():
    """Create cards.data file"""
    content = f"""classes = {PARAMS['classes']}
train = data/train.txt
valid = data/val.txt
names = data/cards.names
backup = backup/
"""
    with open(CONFIG["data_file"], "w") as f:
        f.write(content)

def create_names_file():
    """Create cards.names with all 52 cards"""
    suits = ['h', 'd', 'c', 's']
    ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    
    with open("data/cards.names", "w") as f:
        for rank in ranks:
            for suit in suits:
                f.write(f"{rank}{suit}\n")

def download_weights():
    """Download pretrained weights"""
    if not os.path.exists(CONFIG["pretrained_weights"]):
        print("Downloading pretrained weights...")
        os.system(f"wget https://pjreddie.com/media/files/darknet53.conv.74")

def train():
    """Run training"""
    cmd = [
        CONFIG["darknet_path"],
        "detector", "train",
        CONFIG["data_file"],
        CONFIG["cfg_file"],
        CONFIG["pretrained_weights"],
        "-dont_show", "-map"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    print("=== YOLO Card Training Setup ===")
    setup_directories()
    create_data_file()
    create_names_file()
    download_weights()
    
    print("\nIMPORTANT: Based on your training notes:")
    print("- Step 1: Train to 2000 iterations, check mAP")
    print("- Step 2: If loss oscillates, reduce learning_rate to 0.0001")
    print("- Step 3: Continue to ~2600 iterations for 99.85% mAP")
    
    train() 