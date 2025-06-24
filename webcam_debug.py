import cv2
from ultralytics import YOLO

with open("data/train.txt") as f:
    samples = f.read().splitlines()[:5]

for p in samples:
    img = cv2.imread(p)
    cv2.imshow(f"Train sample: {p}", img)
    cv2.waitKey(0)
cv2.destroyAllWindows()

model = YOLO("best.pt", verbose=False)
print("Class mapping:", model.names)

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    raise RuntimeError("…")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret: break

    frame_count += 1

    # stronger filtering:
    results = model.predict(
        source=frame, conf=0.5, iou=0.45, verbose=False, show=False
    )[0]

    # draw & optionally crop
    for idx, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id, conf  = int(box.cls[0]), float(box.conf[0])
        name = model.names[cls_id]
        label = f"{name} {conf:.2f}"
        cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # show the crop for debugging
        if frame_count % 30 == 0:  # once every 30 frames
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                cv2.imshow(f"Crop_{idx}", crop)

    cv2.imshow("Card Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
