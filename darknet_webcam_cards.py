#!/usr/bin/env python3
"""
Simplified webcam card detection using darknet Python bindings
Based on darknet_video.py example from AlexeyAB/darknet
"""

from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int),
                ("embeddings", POINTER(c_float)),
                ("embedding_size", c_int),
                ("sim", c_float),
                ("track_id", c_int)]

class DETNUMPAIR(Structure):
    _fields_ = [("num", c_int),
                ("dets", POINTER(DETECTION))]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

def network_width(net):
    return lib.network_width(net)

def network_height(net):
    return lib.network_height(net)

def bbox2points(bbox):
    """Convert bbox to corner points"""
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def class_colors(names):
    """Create a color for each class"""
    return [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

def load_network(config_path, weight_path, meta_path):
    """Load darknet network"""
    global metaMain, netMain, altNames
    
    if not os.path.exists(config_path):
        raise ValueError("Invalid config path `" + config_path + "`")
    if not os.path.exists(weight_path):
        raise ValueError("Invalid weight path `" + weight_path + "`")
    if not os.path.exists(meta_path):
        raise ValueError("Invalid data file path `" + meta_path + "`")
    
    if netMain is None:
        netMain = load_net_custom(config_path.encode("ascii"), weight_path.encode("ascii"), 0, 1)
    if metaMain is None:
        metaMain = load_meta(meta_path.encode("ascii"))
    if altNames is None:
        try:
            with open(meta_path) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                    result = result.strip()
                    if result.endswith('.names'):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
        except Exception:
            pass
            
    return netMain, metaMain, altNames

def detect_image(net, meta, im, thresh=.25):
    """Run detection on image"""
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    letter_box = 0
    dets = get_network_boxes(net, im.w, im.h, thresh, 0.5, None, 0, pnum, letter_box)
    num = pnum[0]
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    nameTag = str(i)
                else:
                    nameTag = altNames[i]
                res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_detections(dets, num)
    return res

# Global variables
netMain = None
metaMain = None
altNames = None

# Load the library
lib = CDLL("./darknet/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("./darknet/darknet.dll", RTLD_GLOBAL)  # Windows

lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def cvDrawBoxes(detections, img):
    """Draw bounding boxes on image"""
    for detection in detections:
        x, y, w, h = detection[2][0], detection[2][1], detection[2][2], detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(img, detection[0] + " [" + str(round(detection[1] * 100, 2)) + "%]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

if __name__ == "__main__":
    # Configuration
    configPath = "./cfg/yolov3-cards.cfg"
    weightPath = "./backup/yolov3-cards_best.weights"
    metaPath = "./data/cards.data"
    
    # Update config for 608x608 inference
    inferenceConfig = "./cfg/yolov3-cards-inference.cfg"
    if not os.path.exists(inferenceConfig):
        print("Creating inference config with 608x608 resolution...")
        with open(configPath, 'r') as f:
            cfg = f.read()
        cfg = cfg.replace('width=416', 'width=608')
        cfg = cfg.replace('height=416', 'height=608')
        with open(inferenceConfig, 'w') as f:
            f.write(cfg)
        configPath = inferenceConfig
    
    # Set NMS threshold
    nms = .45
    
    # Check if model exists
    if not os.path.exists(weightPath):
        print(f"Error: Weights not found at {weightPath}")
        print("Please train the model first!")
        exit(1)
    
    # Load network
    print("Loading YOLO network...")
    netMain, metaMain, altNames = load_network(configPath, weightPath, metaPath)
    
    # Create darknet image
    darknet_image = make_image(network_width(netMain), network_height(netMain), 3)
    
    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)  # macOS default webcam
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        print("On macOS, grant camera permission to Terminal/Python")
        exit(1)
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\nWebcam started!")
    print("Press 'q' to quit")
    print("Press 's' to save screenshot\n")
    
    frame_count = 0
    prev_time = time.time()
    
    while True:
        ret, frame_read = cap.read()
        if not ret:
            break
        
        # Calculate FPS
        current_time = time.time()
        fps = 1/(current_time - prev_time)
        prev_time = current_time
        
        # Prepare frame for detection
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (network_width(netMain), network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)
        
        # Copy to darknet image
        copy_image_from_bytes(darknet_image, frame_resized.tobytes())
        
        # Run detection
        detections = detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        
        # Draw results
        image = cvDrawBoxes(detections, frame_read)
        
        # Add FPS counter
        cv2.putText(image, f"FPS: {fps:.1f} | Cards: {len(detections)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Card Detection', image)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"card_detection_{frame_count}.jpg"
            cv2.imwrite(filename, image)
            print(f"Saved: {filename}")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows() 