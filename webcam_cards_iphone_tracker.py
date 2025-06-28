# webcam_cards_iphone_tracker.py - iPhone camera with card state tracking

import cv2
import argparse
from ultralytics import YOLO
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Set, List, Optional
import threading
import os

@dataclass
class CardDetection:
    """Represents a single card detection"""
    card_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    timestamp: float
    center: tuple  # (x, y)

class CardTracker:
    """Tracks card states and handles occlusion/persistence logic"""
    
    def __init__(self, 
                 persistence_frames=30,  # How many frames a card persists after last seen
                 confidence_threshold=0.5,
                 stability_frames=5,     # Frames needed to confirm a new card
                 removal_frames=60):     # Frames needed to confirm card removal
        
        self.persistence_frames = persistence_frames
        self.confidence_threshold = confidence_threshold
        self.stability_frames = stability_frames
        self.removal_frames = removal_frames
        
        # Card state tracking
        self.active_cards: Dict[str, CardDetection] = {}  # Currently visible cards
        self.card_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.card_last_seen: Dict[str, float] = {}
        self.card_first_seen: Dict[str, float] = {}
        self.card_stable_count: Dict[str, int] = defaultdict(int)
        self.card_missing_count: Dict[str, int] = defaultdict(int)
        
        # Game state
        self.played_cards: Set[str] = set()
        self.removed_cards: Set[str] = set()
        self.deck_cards = self._initialize_deck()
        
        # Card counting
        self.running_count = 0
        self.true_count = 0.0
        
        # Statistics
        self.total_detections = 0
        self.frame_count = 0
        
    def _initialize_deck(self) -> Set[str]:
        """Initialize a standard 52-card deck"""
        suits = ['h', 'd', 'c', 's']  # hearts, diamonds, clubs, spades
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        return {f"{rank}{suit}" for rank in ranks for suit in suits}
    
    def _get_card_value(self, card_name: str) -> int:
        """Get Hi-Lo card counting value for a card"""
        # Extract rank from card name (e.g., 'Ah' -> 'A', '10s' -> '10')
        rank = card_name[:-1]  # Remove suit (last character)
        
        # Hi-Lo counting system
        if rank in ['2', '3', '4', '5', '6']:
            return 1  # Low cards: +1
        elif rank in ['7', '8', '9']:
            return 0  # Neutral cards: 0
        elif rank in ['10', 'J', 'Q', 'K', 'A']:
            return -1  # High cards: -1
        else:
            return 0  # Unknown cards: 0
    
    def _update_count(self, card_name: str):
        """Update running count when a new card is detected"""
        card_value = self._get_card_value(card_name)
        self.running_count += card_value
        
        # Calculate true count (running count / decks remaining)
        cards_remaining = len(self.deck_cards - self.played_cards)
        decks_remaining = max(cards_remaining / 52.0, 0.5)  # Minimum 0.5 to avoid division issues
        self.true_count = self.running_count / decks_remaining
    
    def update(self, detections: List[tuple]) -> Dict:
        """Update card tracking with new detections"""
        self.frame_count += 1
        current_time = time.time()
        current_frame_cards = set()
        
        # Process current detections
        for card_name, confidence, bbox in detections:
            if confidence < self.confidence_threshold:
                continue
                
            self.total_detections += 1
            current_frame_cards.add(card_name)
            
            # Calculate center point
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            detection = CardDetection(
                card_name=card_name,
                confidence=confidence,
                bbox=bbox,
                timestamp=current_time,
                center=center
            )
            
            # Update card tracking
            self.card_history[card_name].append(detection)
            self.card_last_seen[card_name] = current_time
            
            # Handle new card logic
            if card_name not in self.active_cards:
                self.card_stable_count[card_name] += 1
                if self.card_stable_count[card_name] >= self.stability_frames:
                    # Card is stable, add to active cards
                    self.active_cards[card_name] = detection
                    if card_name not in self.played_cards:
                        self.card_first_seen[card_name] = current_time
                        self.played_cards.add(card_name)
                        self._update_count(card_name)  # Update card count
                        self._log_card_event("PLAYED", card_name, confidence)
            else:
                # Update existing card
                self.active_cards[card_name] = detection
                self.card_missing_count[card_name] = 0  # Reset missing count
        
        # Handle missing cards (occlusion/removal logic)
        cards_to_remove = []
        for card_name in self.active_cards:
            if card_name not in current_frame_cards:
                self.card_missing_count[card_name] += 1
                
                # Check if card should be considered removed
                if self.card_missing_count[card_name] >= self.removal_frames:
                    cards_to_remove.append(card_name)
                    if card_name not in self.removed_cards:
                        self.removed_cards.add(card_name)
                        # No longer log removal events to keep terminal clean
        
        # Remove cards that have been missing too long
        for card_name in cards_to_remove:
            del self.active_cards[card_name]
            self.card_stable_count[card_name] = 0
        
        # Reset stability count for cards not seen this frame
        for card_name in self.card_stable_count:
            if card_name not in current_frame_cards and card_name not in self.active_cards:
                if self.card_stable_count[card_name] > 0:
                    self.card_stable_count[card_name] = max(0, self.card_stable_count[card_name] - 1)
        
        return self._get_status()
    
    def _log_card_event(self, event_type: str, card_name: str, confidence: float):
        """Log card events to terminal - only for NEW cards"""
        timestamp = time.strftime("%H:%M:%S")
        if event_type == "PLAYED":
            card_value = self._get_card_value(card_name)
            value_str = f"({card_value:+d})" if card_value != 0 else "(0)"
            print(f"\n🃏 [{timestamp}] NEW CARD: {card_name} {value_str} conf: {confidence:.2f}")
            self._print_simple_status()
        # Note: We no longer log REMOVED events to keep output clean
    
    def _print_simple_status(self):
        """Print simplified status for new card detection"""
        print(f"📊 Cards: {len(self.played_cards)}/52 | Running Count: {self.running_count:+d} | True Count: {self.true_count:+.1f}")
        print("-" * 50)
    
    def _print_game_status(self):
        """Print current game status (for detailed view only)"""
        remaining = self.deck_cards - self.played_cards
        print(f"📊 Game Status:")
        print(f"   Cards played: {len(self.played_cards)}/52")
        print(f"   Cards on table: {len(self.active_cards)}")
        print(f"   Cards removed: {len(self.removed_cards)}")
        print(f"   Cards remaining: {len(remaining)}")
        
        if self.active_cards:
            active_list = sorted(list(self.active_cards.keys()))
            print(f"   Currently visible: {', '.join(active_list)}")
        print("-" * 50)
    
    def _get_status(self) -> Dict:
        """Get current tracking status"""
        return {
            'active_cards': dict(self.active_cards),
            'played_cards': self.played_cards.copy(),
            'removed_cards': self.removed_cards.copy(),
            'remaining_cards': self.deck_cards - self.played_cards,
            'frame_count': self.frame_count,
            'total_detections': self.total_detections
        }
    
    def get_card_statistics(self) -> Dict:
        """Get detailed card statistics"""
        stats = {}
        for card_name in self.played_cards:
            history = list(self.card_history[card_name])
            if history:
                confidences = [d.confidence for d in history]
                stats[card_name] = {
                    'detections': len(history),
                    'avg_confidence': sum(confidences) / len(confidences),
                    'max_confidence': max(confidences),
                    'first_seen': self.card_first_seen.get(card_name, 0),
                    'last_seen': self.card_last_seen.get(card_name, 0),
                    'is_active': card_name in self.active_cards,
                    'is_removed': card_name in self.removed_cards
                }
        return stats

def draw_boxes_with_tracking(frame, results, names, tracker: CardTracker):
    """Draw bounding boxes with tracking information"""
    detections = []
    
    # Extract detections
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id, conf = int(box.cls[0]), float(box.conf[0])
        card_name = names[cls_id]
        detections.append((card_name, conf, (x1, y1, x2, y2)))
    
    # Update tracker
    status = tracker.update(detections)
    
    # Draw active cards with enhanced info
    for card_name, detection in tracker.active_cards.items():
        x1, y1, x2, y2 = detection.bbox
        conf = detection.confidence
        
        # Color coding based on card state
        if card_name in tracker.removed_cards:
            color = (0, 0, 255)  # Red for removed
        elif tracker.card_missing_count[card_name] > 0:
            color = (0, 165, 255)  # Orange for temporarily missing
        else:
            color = (0, 255, 0)  # Green for stable
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Enhanced label with tracking info
        missing_count = tracker.card_missing_count[card_name]
        label = f"{card_name}: {conf:.2f}"
        if missing_count > 0:
            label += f" (missing: {missing_count})"
        
        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 4), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add game status overlay
    y_offset = 30
    cv2.putText(frame, f"Cards: {len(status['played_cards'])}/52", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    y_offset += 30
    cv2.putText(frame, f"Running Count: {tracker.running_count:+d}", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    y_offset += 30
    cv2.putText(frame, f"True Count: {tracker.true_count:+.1f}", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    y_offset += 30
    cv2.putText(frame, f"On Table: {len(status['active_cards'])}", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    return frame

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
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Get actual properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✓ iPhone camera initialized: {width}x{height} @ {fps:.1f}fps")
    return cap

def print_final_statistics(tracker: CardTracker):
    """Print final game statistics"""
    print("\n" + "="*60)
    print("FINAL GAME STATISTICS")
    print("="*60)
    
    stats = tracker.get_card_statistics()
    
    print(f"Total cards detected: {len(tracker.played_cards)}/52")
    print(f"Cards currently on table: {len(tracker.active_cards)}")
    print(f"Cards removed from table: {len(tracker.removed_cards)}")
    print(f"Total frames processed: {tracker.frame_count}")
    print(f"Total detections: {tracker.total_detections}")
    
    if stats:
        print(f"\nCard Detection Details:")
        print("-" * 40)
        for card_name in sorted(stats.keys()):
            card_stats = stats[card_name]
            status = "ACTIVE" if card_stats['is_active'] else ("REMOVED" if card_stats['is_removed'] else "PLAYED")
            print(f"{card_name:4s}: {card_stats['detections']:3d} detections, "
                  f"avg conf: {card_stats['avg_confidence']:.2f}, "
                  f"status: {status}")
    
    remaining = tracker.deck_cards - tracker.played_cards
    if remaining:
        print(f"\nCards never seen ({len(remaining)}):")
        remaining_list = sorted(list(remaining))
        for i in range(0, len(remaining_list), 13):  # Print 13 cards per line
            print("  " + ", ".join(remaining_list[i:i+13]))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--confidence", "-c", type=float, default=0.4,
        help="Detection confidence threshold (default: 0.4)"
    )
    p.add_argument(
        "--persistence", "-p", type=int, default=30,
        help="Frames a card persists after last seen (default: 30)"
    )
    p.add_argument(
        "--stability", "-s", type=int, default=5,
        help="Frames needed to confirm new card (default: 5)"
    )
    p.add_argument(
        "--removal", "-r", type=int, default=60,
        help="Frames needed to confirm card removal (default: 60)"
    )
    args = p.parse_args()

    # Load model
    model = YOLO("best_yolo11.pt", verbose=False)
    print("Loaded YOLO model with classes:", model.names)
    
    # Initialize tracker
    tracker = CardTracker(
        persistence_frames=args.persistence,
        confidence_threshold=args.confidence,
        stability_frames=args.stability,
        removal_frames=args.removal
    )
    
    # Setup camera
    cap = setup_iphone_camera()
    
    print("\n" + "="*60)
    print("CARD TRACKING SYSTEM STARTED")
    print("="*60)
    print("🎯 DETECTION MODE: Only NEW cards will be reported to terminal")
    print("   (Cards already detected will be ignored)")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("  'r' - Reset game state")
    print("  't' - Show detailed statistics")
    print(f"\nSettings:")
    print(f"  Confidence threshold: {args.confidence}")
    print(f"  Stability frames: {args.stability}")
    print(f"  Removal frames: {args.removal}")
    print("-" * 60)

    screenshot_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from iPhone camera")
                break

            # Run inference
            results = model.predict(
                source=frame, conf=args.confidence, iou=0.45, verbose=False, show=False
            )[0]

            # Draw with tracking
            annotated = draw_boxes_with_tracking(frame, results, model.names, tracker)
            
            cv2.imshow("iPhone Card Tracker", annotated)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"card_tracker_screenshot_{screenshot_count}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"Screenshot saved: {filename}")
            elif key == ord('r'):
                tracker = CardTracker(
                    persistence_frames=args.persistence,
                    confidence_threshold=args.confidence,
                    stability_frames=args.stability,
                    removal_frames=args.removal
                )
                print("\n🔄 Game state reset! Count reset to 0.")
            elif key == ord('t'):
                print_final_statistics(tracker)

    except KeyboardInterrupt:
        print("\nStopping card tracker...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print_final_statistics(tracker) 