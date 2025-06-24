# webcam_cards.py

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

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--image", "-i", type=str,
        help="Path to an image file (skip webcam and annotate this)"
    )
    p.add_argument(
        "--output", "-o", type=str, default="annotated.jpg",
        help="Where to save the annotated image in --image mode"
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
            source=img, conf=0.5, iou=0.45, verbose=False, show=False
        )[0]

        print_detections(results, model.names, context="Image mode: ")
        annotated = draw_boxes(img.copy(), results, model.names)

        cv2.imwrite(args.output, annotated)
        print(f"Annotated image written to: {args.output}")
        cv2.imshow("Annotated Image", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        # ----- WEBCAM MODE -----
        # Try iPhone camera first (index 1), fallback to built-in camera (index 0)
        camera_indices = [1, 0]  # iPhone first, then built-in camera
        cap = None
        
        for cam_idx in camera_indices:
            print(f"Trying camera {cam_idx}...")
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✓ Successfully opened camera {cam_idx}")
                    break
                else:
                    cap.release()
                    cap = None
            else:
                if cap:
                    cap.release()
                cap = None
        
        if cap is None or not cap.isOpened():
            raise RuntimeError(
                "Could not open any camera. Make sure:\n"
                "1. Your iPhone is connected via USB\n"
                "2. You've trusted this computer on your iPhone\n"
                "3. Camera access is granted in System Preferences → Security & Privacy → Camera"
            )
        print("Webcam opened, starting inference...")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            results = model.predict(
                source=frame, conf=0.5, iou=0.45, verbose=False, show=False
            )[0]

            print_detections(results, model.names, context=f"[Frame {frame_count}] ")

            annotated = draw_boxes(frame, results, model.names)
            cv2.imshow("Card Detection", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
