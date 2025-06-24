#!/usr/bin/env python3
"""
Real-time playing card detection using macOS webcam
Based on AlexeyAB/darknet YOLO implementation
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Import darknet library
sys.path.append('./darknet')
sys.path.append('./darknet/python')

try:
    import darknet
except ImportError:
    print("Error: Could not import darknet. Make sure darknet is built and in the path.")
    print("Try: export PYTHONPATH=$PYTHONPATH:./darknet/python")
    sys.exit(1)

class CardDetector:
    def __init__(self, config_path, weights_path, data_path):
        """Initialize YOLO detector for playing cards"""
        
        # Check if model files exist
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load class names
        self.class_names = self.load_class_names(data_path)
        
        # Initialize darknet network
        print("Loading YOLO network...")
        self.network, self.class_names_darknet, self.class_colors = darknet.load_network(
            config_path,
            data_path,
            weights_path,
            batch_size=1
        )
        
        # Get network dimensions
        self.darknet_width = darknet.network_width(self.network)
        self.darknet_height = darknet.network_height(self.network)
        
        print(f"Network loaded. Input size: {self.darknet_width}x{self.darknet_height}")
        print(f"Detecting {len(self.class_names)} classes: {len(self.class_names)} playing cards")
        
        # Create darknet image
        self.darknet_image = darknet.make_image(self.darknet_width, self.darknet_height, 3)
    
    def load_class_names(self, data_path):
        """Load class names from data file"""
        names_file = None
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip().startswith('names'):
                    names_file = line.split('=')[1].strip()
                    break
        
        if not names_file or not os.path.exists(names_file):
            # Default card names if file not found
            suits = ['h', 'd', 'c', 's']
            ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
            return [f"{rank}{suit}" for rank in ranks for suit in suits]
        
        with open(names_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    def detect_cards(self, frame, threshold=0.25):
        """Detect cards in a frame"""
        # Resize frame to network dimensions
        frame_resized = cv2.resize(frame, (self.darknet_width, self.darknet_height),
                                   interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Copy to darknet image
        darknet.copy_image_from_bytes(self.darknet_image, frame_rgb.tobytes())
        
        # Run detection
        detections = darknet.detect_image(self.network, self.class_names_darknet, 
                                        self.darknet_image, thresh=threshold)
        
        # Convert coordinates back to original frame size
        height, width = frame.shape[:2]
        x_scale = width / self.darknet_width
        y_scale = height / self.darknet_height
        
        scaled_detections = []
        for label, confidence, bbox in detections:
            x, y, w, h = bbox
            x *= x_scale
            y *= y_scale
            w *= x_scale
            h *= y_scale
            scaled_detections.append((label, confidence, (x, y, w, h)))
        
        return scaled_detections
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        for label, confidence, bbox in detections:
            x, y, w, h = bbox
            x1 = int(x - w/2)
            y1 = int(y - h/2)
            x2 = int(x + w/2)
            y2 = int(y + h/2)
            
            # Get color for this class
            class_id = self.class_names.index(label) if label in self.class_names else 0
            color = self.class_colors[class_id]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence
            label_text = f"{label}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw label background
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 4), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label_text, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

def main():
    """Main function to run webcam detection"""
    
    # Configuration
    CONFIG = {
        "cfg": "cfg/yolov3-cards.cfg",  # Use 608x608 for better accuracy as mentioned
        "weights": "backup/yolov3-cards_best.weights",  # Use best weights
        "data": "data/cards.data",
        "threshold": 0.25,  # Detection threshold
        "webcam_id": 0  # Default macOS webcam
    }
    
    # Update config for inference (higher resolution)
    inference_cfg = "cfg/yolov3-cards-inference.cfg"
    if not os.path.exists(inference_cfg):
        print("Creating inference configuration with 608x608 resolution...")
        with open(CONFIG["cfg"], 'r') as f:
            cfg_content = f.read()
        
        # Update resolution for better accuracy
        cfg_content = cfg_content.replace('width=416', 'width=608')
        cfg_content = cfg_content.replace('height=416', 'height=608')
        
        with open(inference_cfg, 'w') as f:
            f.write(cfg_content)
        
        CONFIG["cfg"] = inference_cfg
    
    # Check if model is trained
    if not os.path.exists(CONFIG["weights"]):
        print(f"Error: Trained weights not found at {CONFIG['weights']}")
        print("Please train the model first using train_cards_model.py")
        sys.exit(1)
    
    # Initialize detector
    try:
        detector = CardDetector(CONFIG["cfg"], CONFIG["weights"], CONFIG["data"])
    except Exception as e:
        print(f"Error initializing detector: {e}")
        sys.exit(1)
    
    # Open webcam
    print(f"Opening webcam (ID: {CONFIG['webcam_id']})...")
    cap = cv2.VideoCapture(CONFIG["webcam_id"])
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        print("On macOS, you may need to grant camera permissions to Terminal/Python")
        sys.exit(1)
    
    # Set webcam properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\nWebcam started. Press 'q' to quit, 's' to save screenshot")
    print(f"Detection threshold: {CONFIG['threshold']}")
    print("Press '+' to increase threshold, '-' to decrease\n")
    
    threshold = CONFIG["threshold"]
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from webcam")
            break
        
        # Detect cards
        detections = detector.detect_cards(frame, threshold)
        
        # Draw detections
        frame = detector.draw_detections(frame, detections)
        
        # Display statistics
        stats_text = f"Cards detected: {len(detections)} | Threshold: {threshold:.2f} | FPS: {cap.get(cv2.CAP_PROP_FPS):.1f}"
        cv2.putText(frame, stats_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("Playing Card Detection", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"card_detection_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
        elif key == ord('+'):
            threshold = min(threshold + 0.05, 1.0)
            print(f"Threshold increased to: {threshold:.2f}")
        elif key == ord('-'):
            threshold = max(threshold - 0.05, 0.1)
            print(f"Threshold decreased to: {threshold:.2f}")
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    darknet.free_image(detector.darknet_image)
    print("\nWebcam detection stopped.")

if __name__ == "__main__":
    print("=== YOLO Playing Card Detection (Webcam) ===")
    print("Make sure you have:")
    print("1. Trained the model using train_cards_model.py")
    print("2. Built darknet with OpenCV support")
    print("3. Granted camera permissions on macOS")
    print("="*50 + "\n")
    
    main() 