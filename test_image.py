#!/usr/bin/env python3
"""
Simple script to test best_yolo11.pt model on IMG_8515.jpeg
"""

import cv2
from ultralytics import YOLO
import argparse

def draw_detections(image, results, model_names, confidence_threshold=0.4):
    """Draw bounding boxes and labels on the image"""
    annotated_image = image.copy()
    
    # Extract detections
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id, conf = int(box.cls[0]), float(box.conf[0])
        
        if conf < confidence_threshold:
            continue
            
        card_name = model_names[cls_id]
        
        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label with confidence
        label = f"{card_name}: {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 4), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(annotated_image, label, (x1, y1 - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return annotated_image

def main():
    parser = argparse.ArgumentParser(description='Test YOLO model on image')
    parser.add_argument('--model', '-m', default='best_yolo11.pt', 
                       help='Path to YOLO model (default: best_yolo11.pt)')
    parser.add_argument('--image', '-i', default='IMG_8515.jpeg', 
                       help='Path to image (default: IMG_8515.jpeg)')
    parser.add_argument('--confidence', '-c', type=float, default=0.4, 
                       help='Confidence threshold (default: 0.4)')
    parser.add_argument('--output', '-o', default='output_detection.jpg', 
                       help='Output image path (default: output_detection.jpg)')
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model, verbose=False)
    print(f"Model loaded with {len(model.names)} classes: {list(model.names.values())}")
    
    # Load image
    print(f"Loading image: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image {args.image}")
        return
    
    print(f"Image loaded: {image.shape[1]}x{image.shape[0]}")
    
    # Run inference
    print(f"Running inference with confidence threshold: {args.confidence}")
    results = model.predict(
        source=image, 
        conf=args.confidence, 
        iou=0.45, 
        verbose=False, 
        show=False
    )[0]
    
    # Print detections
    num_detections = len(results.boxes) if results.boxes is not None else 0
    print(f"\nDetections found: {num_detections}")
    
    if num_detections > 0:
        print("\nDetected cards:")
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id, conf = int(box.cls[0]), float(box.conf[0])
            card_name = model.names[cls_id]
            print(f"  {i+1}. {card_name}: {conf:.3f} at [{x1}, {y1}, {x2}, {y2}]")
    
    # Draw detections and save result
    annotated_image = draw_detections(image, results, model.names, args.confidence)
    cv2.imwrite(args.output, annotated_image)
    print(f"\nAnnotated image saved as: {args.output}")
    
    # Display image (optional - will only work if display is available)
    try:
        cv2.imshow("Card Detection", annotated_image)
        print("Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("Display not available, skipping image preview")

if __name__ == "__main__":
    main() 