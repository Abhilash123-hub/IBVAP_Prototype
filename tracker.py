"""
Border Object Tracker (IBVAP)
Defense-Grade Multi-Object Tracking implementing the ByteTrack Two-Stage Association Algorithm.
Tracks humans, vehicles, and calculates kinematics: Direction, Speed, Duration, and Path History.
Recovers occluded and crawling targets using low-confidence detection association.
"""

import time
import math
import numpy as np


class TrackedObject:
    def __init__(self, track_id, box, class_id, label, confidence):
        self.track_id = track_id
        self.box = list(map(float, box))
        self.class_id = class_id
        self.label = label
        self.confidence = confidence
        
        self.start_time = time.time()
        self.last_seen = time.time()
        self.hits = 1
        self.misses = 0
        
        cx, cy = self._get_centroid(self.box)
        self.history = [(cx, cy)]  # list of (x, y) tuples for path tracking
        self.velocity = (0.0, 0.0)
        self.speed = 0.0
        self.direction = "Stationary"
        self.is_loitering = False
        self.is_crawling = False
        self.is_sprinting = False

        # Post-Fence Breach Telemetry (Tracks subject after crossing into domestic territory)
        self.has_crossed_fence = False
        self.fence_cross_time = None
        self.fence_cross_pos = None
        self.penetration_depth = 0.0  # Distance past fence into secure zone (pixels)
        self.post_cross_dwell = 0.0   # Dwell time after crossing fence (seconds)
        self.post_breach_path = []

    @staticmethod
    def _get_centroid(box):
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def mark_fence_breached(self, cross_point=None, cross_time=None):
        """Permanently seals this track as having breached the virtual fence."""
        if not self.has_crossed_fence:
            self.has_crossed_fence = True
            self.fence_cross_time = cross_time or time.time()
            if cross_point:
                self.fence_cross_pos = cross_point
            elif len(self.history) > 0:
                self.fence_cross_pos = self.history[-1]

    def update_post_breach_telemetry(self, current_pos=None, fence_ref_y=None):
        """Updates penetration depth and time elapsed since crossing the virtual fence."""
        if not self.has_crossed_fence:
            return

        now = time.time()
        if self.fence_cross_time is not None:
            self.post_cross_dwell = round(now - self.fence_cross_time, 1)

        pos = current_pos or (self.history[-1] if self.history else None)
        if pos:
            self.post_breach_path.append(pos)
            if len(self.post_breach_path) > 40:
                self.post_breach_path.pop(0)

            # If fence reference Y is given, penetration is vertical distance past fence into territory
            if fence_ref_y is not None:
                self.penetration_depth = max(0.0, float(pos[1] - fence_ref_y))
            elif self.fence_cross_pos is not None:
                # Euclidean distance from breach point
                dx = pos[0] - self.fence_cross_pos[0]
                dy = pos[1] - self.fence_cross_pos[1]
                self.penetration_depth = round(math.sqrt(dx ** 2 + dy ** 2), 1)

    def update(self, box, confidence):
        prev_cx, prev_cy = self._get_centroid(self.box)
        self.box = list(map(float, box))
        self.confidence = confidence
        now = time.time()
        dt = max(now - self.last_seen, 0.001)
        self.last_seen = now
        self.hits += 1
        self.misses = 0

        new_cx, new_cy = self._get_centroid(self.box)
        self.history.append((new_cx, new_cy))
        if len(self.history) > 40:
            self.history.pop(0)

        # Compute velocity (pixels/second)
        vx = (new_cx - prev_cx) / dt
        vy = (new_cy - prev_cy) / dt
        # Smooth velocity with exponential moving average
        self.velocity = (0.7 * self.velocity[0] + 0.3 * vx, 0.7 * self.velocity[1] + 0.3 * vy)
        self.speed = math.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)

        # Direction analysis
        if self.speed < 15:
            self.direction = "Stationary"
        elif abs(vy) > abs(vx):
            self.direction = "Inbound (South)" if vy > 0 else "Outbound (North)"
        else:
            self.direction = "Eastbound" if vx > 0 else "Westbound"

        if self.has_crossed_fence:
            self.update_post_breach_telemetry(current_pos=(new_cx, new_cy))

    @property
    def dwell_time(self):
        """Dwell duration in seconds."""
        return round(time.time() - self.start_time, 1)

    @property
    def duration(self):
        """Alias for dwell duration."""
        return self.dwell_time

    @property
    def path(self):
        """Historical coordinate points [(x, y), ...] showing path taken."""
        return self.history

    @property
    def aspect_ratio(self):
        """Width / Height ratio. Persons usually have W/H < 0.6. Prone/crawling has W/H >= 0.95."""
        w = max(self.box[2] - self.box[0], 1)
        h = max(self.box[3] - self.box[1], 1)
        return round(w / h, 2)


class BorderObjectTracker:
    """
    ByteTrack-based Multi-Object Tracker.
    Associates high-confidence detections first, then recovers occluded/crawling
    targets using low-confidence detections in a secondary matching pass.
    """
    def __init__(self, high_thresh=0.40, low_thresh=0.15, iou_threshold=0.25, max_distance=95, max_misses=15, max_lost=None, **kwargs):
        self.tracks = {}
        self.next_id = 1
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        self.max_misses = max_lost if max_lost is not None else max_misses

    @staticmethod
    def compute_iou(box1, box2):
        x1_a, y1_a, x2_a, y2_a = box1
        x1_b, y1_b, x2_b, y2_b = box2

        x_left = max(x1_a, x1_b)
        y_top = max(y1_a, y1_b)
        x_right = min(x2_a, x2_b)
        y_bottom = min(y2_a, y2_b)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (x2_a - x1_a) * (y2_a - y1_a)
        box2_area = (x2_b - x1_b) * (y2_b - y1_b)
        union_area = float(box1_area + box2_area - intersection_area)

        if union_area <= 0:
            return 0.0
        return intersection_area / union_area

    @staticmethod
    def euclidean_distance(box1, box2):
        cx1 = (box1[0] + box1[2]) / 2.0
        cy1 = (box1[1] + box1[3]) / 2.0
        cx2 = (box2[0] + box2[2]) / 2.0
        cy2 = (box2[1] + box2[3]) / 2.0
        return math.hypot(cx1 - cx2, cy1 - cy2)

    def update(self, detections):
        """
        ByteTrack Two-Stage Association Algorithm:
        1. Separate detections into high-score (>= high_thresh) and low-score (low_thresh <= conf < high_thresh).
        2. First association: match existing tracks with high-score detections.
        3. Second association: match remaining unmatched tracks with low-score detections
           (recovers occluded, crawling, or partially obscured targets, preventing track loss).
        4. Initialize new tracks only from remaining high-score detections.
        5. Remove lost tracks exceeding max_misses.
        """
        # Split detections
        det_high = []
        det_low = []
        for det in detections:
            conf = det.get("confidence", 0.5)
            if conf >= self.high_thresh:
                det_high.append(det)
            elif conf >= self.low_thresh:
                det_low.append(det)

        active_track_ids = list(self.tracks.keys())
        matched_track_ids = set()
        unmatched_high_dets = []

        # --- FIRST ASSOCIATION: Match Existing Tracks with High-Score Detections ---
        for det in det_high:
            best_id = None
            best_iou = self.iou_threshold
            best_dist = self.max_distance

            for tid in active_track_ids:
                if tid in matched_track_ids:
                    continue
                tr = self.tracks[tid]
                # Same general class group (Person with Person, Vehicle with Vehicle)
                if (tr.label == "Person" and det["label"] != "Person") or (tr.label != "Person" and det["label"] == "Person"):
                    continue

                iou = self.compute_iou(tr.box, det["box"])
                dist = self.euclidean_distance(tr.box, det["box"])

                if iou >= best_iou or (dist < self.max_distance and dist < best_dist):
                    best_id = tid
                    best_iou = iou
                    best_dist = dist

            if best_id is not None:
                matched_track_ids.add(best_id)
                self.tracks[best_id].update(det["box"], det["confidence"])
            else:
                unmatched_high_dets.append(det)

        # --- SECOND ASSOCIATION (ByteTrack Innovation): Match Unmatched Tracks with Low-Score Detections ---
        unmatched_track_ids = [tid for tid in active_track_ids if tid not in matched_track_ids]
        for det in det_low:
            best_id = None
            best_iou = self.iou_threshold * 0.75
            best_dist = self.max_distance * 1.15

            for tid in unmatched_track_ids:
                if tid in matched_track_ids:
                    continue
                tr = self.tracks[tid]
                if (tr.label == "Person" and det["label"] != "Person") or (tr.label != "Person" and det["label"] == "Person"):
                    continue

                iou = self.compute_iou(tr.box, det["box"])
                dist = self.euclidean_distance(tr.box, det["box"])

                if iou >= best_iou or (dist < self.max_distance * 1.15 and dist < best_dist):
                    best_id = tid
                    best_iou = iou
                    best_dist = dist

            if best_id is not None:
                matched_track_ids.add(best_id)
                # Recover track using low-score detection (crawling/occlusion recovery)
                self.tracks[best_id].update(det["box"], det["confidence"])

        # Mark missed tracks
        dead_ids = []
        for tid, tr in self.tracks.items():
            if tid not in matched_track_ids:
                tr.misses += 1
                if tr.misses > self.max_misses:
                    dead_ids.append(tid)

        for tid in dead_ids:
            del self.tracks[tid]

        # Initialize new tracks ONLY from remaining high-score detections
        for det in unmatched_high_dets:
            new_track = TrackedObject(
                track_id=self.next_id,
                box=det["box"],
                class_id=det["class_id"],
                label=det["label"],
                confidence=det["confidence"]
            )
            self.tracks[self.next_id] = new_track
            self.next_id += 1

        return list(self.tracks.values())


# ByteTrack Alias
ByteTrackTracker = BorderObjectTracker
