#!/usr/bin/env python3
"""
Train YOLO model for playing card detection
Based on AlexeyAB/darknet implementation
"""

import os
import subprocess
import sys
from pathlib import Path

# Training configuration based on your image
CONFIG = {
    "darknet_path": "./darknet",  # Path to darknet executable
    "cfg_file": "cfg/yolov3-cards.cfg",  # Your custom config file
    "data_file": "data/cards.data",  # Your data configuration
    "pretrained_weights": "darknet53.conv.74",  # Pre-trained weights for transfer learning
    "output_dir": "backup/",
    "final_weights": "backup/yolov3-cards_final.weights"
}

# Training parameters from your image
TRAINING_PARAMS = {
    "batch": 64,
    "subdivisions": 16,
    "width": 416,
    "height": 416,
    "flip": 0,  # Important: no random flips for cards
    "learning_rate": 0.001,
    "classes": 52,  # 52 playing cards
    "filters": 171,  # (classes + 5) * 3 = (52 + 5) * 3 = 171
    "random": 1,  # Train for different resolutions
}

def create_cfg_file():
    """Create or modify the cfg file with your parameters"""
    print("Creating/updating configuration file...")
    
    # Base YOLOv3 configuration template
    cfg_content = f"""[net]
# Training
batch={TRAINING_PARAMS['batch']}
subdivisions={TRAINING_PARAMS['subdivisions']}
width={TRAINING_PARAMS['width']}
height={TRAINING_PARAMS['height']}
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate={TRAINING_PARAMS['learning_rate']}
burn_in=1000
max_batches = 104000
policy=steps
steps=83200,93600
scales=.1,.1

flip={TRAINING_PARAMS['flip']}  # No flipping for playing cards!

# Add the full YOLOv3 architecture here...
# For brevity, I'm showing the key modifications for [yolo] layers

[yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes={TRAINING_PARAMS['classes']}
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random={TRAINING_PARAMS['random']}

# Before each [yolo] layer, update filters
[convolutional]
size=1
stride=1
pad=1
filters={TRAINING_PARAMS['filters']}
activation=linear
"""
    
    os.makedirs("cfg", exist_ok=True)
    with open(CONFIG["cfg_file"], "w") as f:
        f.write(cfg_content)
    print(f"Configuration saved to {CONFIG['cfg_file']}")

def create_data_file():
    """Create the data configuration file"""
    print("Creating data configuration file...")
    
    data_content = f"""classes = {TRAINING_PARAMS['classes']}
train = data/train.txt
valid = data/val.txt
names = data/cards.names
backup = {CONFIG['output_dir']}
"""
    
    os.makedirs("data", exist_ok=True)
    with open(CONFIG["data_file"], "w") as f:
        f.write(data_content)
    print(f"Data configuration saved to {CONFIG['data_file']}")

def create_names_file():
    """Create cards.names file with all 52 playing cards"""
    print("Creating names file...")
    
    suits = ['h', 'd', 'c', 's']  # hearts, diamonds, clubs, spades
    ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    
    card_names = []
    for rank in ranks:
        for suit in suits:
            card_names.append(f"{rank}{suit}")
    
    with open("data/cards.names", "w") as f:
        f.write("\n".join(card_names))
    print(f"Created {len(card_names)} card names")

def download_pretrained_weights():
    """Download pre-trained weights for transfer learning"""
    if not os.path.exists(CONFIG["pretrained_weights"]):
        print("Downloading pre-trained weights...")
        url = "https://pjreddie.com/media/files/darknet53.conv.74"
        subprocess.run(["wget", "-O", CONFIG["pretrained_weights"], url])
    else:
        print("Pre-trained weights already exist")

def train_model(resume_from=None):
    """Train the YOLO model"""
    print("\nStarting training...")
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    cmd = [
        CONFIG["darknet_path"],
        "detector", "train",
        CONFIG["data_file"],
        CONFIG["cfg_file"]
    ]
    
    if resume_from:
        cmd.append(resume_from)
    else:
        cmd.append(CONFIG["pretrained_weights"])
    
    # Add flags
    cmd.extend(["-dont_show", "-mjpeg_port", "8090", "-map"])
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Monitor training progress
        for line in process.stdout:
            print(line.strip())
            
            # Check for specific iterations mentioned in your training steps
            if "avg loss" in line:
                if "2000:" in line:
                    print("\n>>> Step 1 complete! Check mAP. If loss oscillates, reduce learning rate.")
                elif "2600:" in line:
                    print("\n>>> Step 2 checkpoint! mAP should be ~99.85%")
                elif "4500:" in line:
                    print("\n>>> Step 3 complete! Training finished when loss stops improving.")
        
        process.wait()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        process.terminate()
    except Exception as e:
        print(f"Error during training: {e}")

def main():
    print("=== YOLO Playing Cards Training Script ===")
    print(f"Training for {TRAINING_PARAMS['classes']} classes (52 playing cards)")
    
    # Setup
    create_cfg_file()
    create_data_file()
    create_names_file()
    download_pretrained_weights()
    
    print("\n" + "="*50)
    print("IMPORTANT: Based on your training image:")
    print("- Step 1: Train until iteration 2000, check if mAP=68")
    print("- Step 2: If loss oscillates, reduce learning_rate to 0.0001")
    print("- Step 3: Continue training until ~2600 iterations (mAP should be 99.85%)")
    print("- Step 4: For inference, increase resolution to 608x608")
    print("="*50 + "\n")
    
    # Check if we should resume training
    resume_weights = None
    if os.path.exists(f"{CONFIG['output_dir']}yolov3-cards_last.weights"):
        response = input("Found existing weights. Resume training? (y/n): ")
        if response.lower() == 'y':
            resume_weights = f"{CONFIG['output_dir']}yolov3-cards_last.weights"
    
    # Train
    train_model(resume_from=resume_weights)
    
    print("\nTraining complete!")
    print(f"Weights saved in: {CONFIG['output_dir']}")
    print(f"Best weights: {CONFIG['output_dir']}yolov3-cards_best.weights")
    print(f"Final weights: {CONFIG['output_dir']}yolov3-cards_final.weights")

if __name__ == "__main__":
    # Check if darknet exists
    if not os.path.exists(CONFIG["darknet_path"]):
        print("ERROR: Darknet not found!")
        print("Please clone and build darknet first:")
        print("  git clone https://github.com/AlexeyAB/darknet")
        print("  cd darknet")
        print("  make")
        sys.exit(1)
    
    main() 