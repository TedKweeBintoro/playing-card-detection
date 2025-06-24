# webcam_cards_iphone.py - Optimized for iPhone camera via USB

import cv2
import argparse
from ultralytics import YOLO

def draw_boxes(frame, results, names):
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id, conf  = int(box.cls[0]), float(box.conf[0])
        label         = f"{names[cls_id]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return frame

def print_detections(results, names, context=""):
    if not results.boxes:
        print(f"{context}No cards detected.")
    else:
        print(f"{context}Detected cards:")
        for box in results.boxes:
            cls_id, conf = int(box.cls[0]), float(box.conf[0])
            print(f"  - {names[cls_id]} @ {conf:.2f}")

def setup_iphone_camera():
    """Setup iPhone camera with optimal settings"""
    print("Setting up iPhone camera...")
    
    # Try iPhone camera (usually index 1)
    cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
    
    if not cap.isOpened():
        print("iPhone camera not found on index 1, trying index 0...")
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open iPhone camera. Make sure:\n"
            "1. iPhone is connected via USB\n"
            "2. You've trusted this computer on your iPhone\n"
            "3. Camera permissions are granted in System Preferences"
        )
    
    # Set optimal properties for iPhone camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Try to increase FPS
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
    
    # Get actual properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✓ iPhone camera initialized: {width}x{height} @ {fps:.1f}fps")
    return cap

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--image", "-i", type=str,
        help="Path to an image file (skip webcam and annotate this)"
    )
    p.add_argument(
        "--output", "-o", type=str, default="annotated_iphone.jpg",
        help="Where to save the annotated image in --image mode"
    )
    p.add_argument(
        "--confidence", "-c", type=float, default=0.5,
        help="Detection confidence threshold (default: 0.5)"
    )
    args = p.parse_args()

    # Load model once
    model = YOLO("best_jun21.pt", verbose=False)
    print("Class mapping:", model.names)

    if args.image:
        # ----- IMAGE MODE -----
        img = cv2.imread(args.image)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {args.image}")

        results = model.predict(
            source=img, conf=args.confidence, iou=0.45, verbose=False, show=False
        )[0]

        print_detections(results, model.names, context="Image mode: ")
        annotated = draw_boxes(img.copy(), results, model.names)

        cv2.imwrite(args.output, annotated)
        print(f"Annotated image written to: {args.output}")
        cv2.imshow("Annotated Image", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        # ----- IPHONE WEBCAM MODE -----
        cap = setup_iphone_camera()
        
        print("iPhone camera ready! Starting card detection...")
        print("Press 'q' to quit, 's' to save screenshot")
        print(f"Detection confidence: {args.confidence}")

        frame_count = 0
        screenshot_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from iPhone camera")
                    break
                    
                frame_count += 1

                # Run inference
                results = model.predict(
                    source=frame, conf=args.confidence, iou=0.45, verbose=False, show=False
                )[0]

                # Print detections every 30 frames to reduce spam
                if frame_count % 30 == 0:
                    print_detections(results, model.names, context=f"[Frame {frame_count}] ")

                # Draw annotations
                annotated = draw_boxes(frame, results, model.names)
                
                # Add frame info
                cv2.putText(annotated, f"iPhone Camera - Frame {frame_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow("iPhone Card Detection", annotated)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_count += 1
                    filename = f"iphone_screenshot_{screenshot_count}.jpg"
                    cv2.imwrite(filename, annotated)
                    print(f"Screenshot saved: {filename}")

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"Processed {frame_count} frames from iPhone camera") 