#!/usr/bin/env python3
"""
Real-time playing card detection using macOS webcam
Uses AlexeyAB/darknet Python bindings
"""

import cv2
import os
import sys

# Add darknet to Python path
sys.path.insert(0, './darknet')
sys.path.insert(0, './darknet/python')

try:
    import darknet
except ImportError:
    print("Error: Could not import darknet")
    print("Make sure darknet is built with: make")
    print("And Python bindings are available")
    sys.exit(1)

def main():
    # Configuration
    config_file = "cfg/yolov3-cards.cfg"
    weights_file = "backup/yolov3-cards_best.weights"
    data_file = "data/cards.data"
    thresh = 0.25
    
    # For inference, use 608x608 (as mentioned in your training notes)
    inference_cfg = "cfg/yolov3-cards-608.cfg"
    if not os.path.exists(inference_cfg) and os.path.exists(config_file):
        print("Creating 608x608 inference config...")
        with open(config_file, 'r') as f:
            cfg = f.read()
        cfg = cfg.replace('width=416', 'width=608')
        cfg = cfg.replace('height=416', 'height=608')
        with open(inference_cfg, 'w') as f:
            f.write(cfg)
        config_file = inference_cfg
    
    # Check if weights exist
    if not os.path.exists(weights_file):
        print(f"Error: Weights not found at {weights_file}")
        print("Train the model first using train_yolo_cards.py")
        sys.exit(1)
    
    # Load network
    print("Loading YOLO network...")
    network, class_names, class_colors = darknet.load_network(
        config_file,
        data_file,
        weights_file,
        batch_size=1
    )
    
    # Get network size
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    print(f"Network size: {width}x{height}")
    
    # Create darknet image
    darknet_image = darknet.make_image(width, height, 3)
    
    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        print("On macOS, grant camera permissions")
        sys.exit(1)
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\nPress 'q' to quit, 's' to save screenshot")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Prepare frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        
        # Copy to darknet
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
        
        # Detect
        detections = darknet.detect_image(network, class_names, darknet_image, thresh)
        
        # Draw boxes
        for label, confidence, bbox in detections:
            x, y, w, h = bbox
            # Convert to frame coordinates
            x = x * frame.shape[1] / width
            y = y * frame.shape[0] / height
            w = w * frame.shape[1] / width
            h = h * frame.shape[0] / height
            
            # Draw box
            x1 = int(x - w/2)
            y1 = int(y - h/2)
            x2 = int(x + w/2)
            y2 = int(y + h/2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", 
                       (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)
        
        # Show stats
        cv2.putText(frame, f"Cards: {len(detections)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display
        cv2.imshow('Card Detection', frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"detection_{frame_count}.jpg", frame)
            print(f"Saved detection_{frame_count}.jpg")
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("=== Playing Card Detection (Webcam) ===")
    main() 