#!/usr/bin/env python3
"""
Monitor YOLO training progress
"""

import subprocess
import time
import os

def check_training():
    """Check if training is running"""
    result = subprocess.run(['pgrep', '-f', 'darknet.*detector.*train'], 
                          capture_output=True, text=True)
    return result.returncode == 0

def get_latest_weights():
    """Get the latest weights file"""
    backup_files = []
    if os.path.exists('backup'):
        for f in os.listdir('backup'):
            if f.endswith('.weights'):
                backup_files.append(f)
    return sorted(backup_files)

def main():
    print("=== YOLO Training Monitor ===")
    print("Press Ctrl+C to stop monitoring\n")
    
    while True:
        # Check if training is running
        if check_training():
            print(f"\r✓ Training is running", end='', flush=True)
        else:
            print(f"\r✗ Training is not running", end='', flush=True)
        
        # Check for weight files
        weights = get_latest_weights()
        if weights:
            print(f" | Latest checkpoint: {weights[-1]}", end='', flush=True)
        else:
            print(f" | No checkpoints yet", end='', flush=True)
        
        time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.") 