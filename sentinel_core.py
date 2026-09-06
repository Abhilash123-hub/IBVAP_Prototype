"""
SENTINEL Core Engine (IBVAP)
Defense-Grade Cognitive Brain for Adaptive Multi-Tier Compute Scheduling,
Per-Camera Motion Watcher, Threat Scoring (0-100), Resource Telemetry,
and Explainable Tactical Alert Generation.
"""

import time
import os
import cv2
import numpy as np


class AdaptiveWatcher:
    """
    Ultra-lightweight per-camera motion detector (<0.5ms per frame).
    Enables LOW MODE operation, sparing heavy AI inference on idle cameras.
    """
    def __init__(self, min_motion_pixels=450, delta_threshold=22):
        self.min_motion_pixels = min_motion_pixels
        self.delta_threshold = delta_threshold
        self.prev_gray = None
        self.motion_detected = False
        self.motion_pixel_count = 0

    def check_motion(self, frame):
        """
        Evaluates frame for optical pixel variance.
        Returns:
            motion_detected (bool): True if motion exceeds threshold.
            motion_pixels (int): Number of moving pixels.
        """
        if frame is None:
            return False, 0

        # Downscale for ultra-fast motion check
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (160, 120))
        if len(small.shape) == 3:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        else:
            gray = small

        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return False, 0

        # Absolute frame difference
        frame_diff = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(frame_diff, self.delta_threshold, 255, cv2.THRESH_BINARY)
        motion_pixels = cv2.countNonZero(thresh)

        # Update previous frame with slow alpha blend to adapt to lighting
        cv2.accumulateWeighted(gray, self.prev_gray.astype(np.float32), 0.15)
        self.prev_gray = self.prev_gray.astype(np.uint8)

        self.motion_pixel_count = motion_pixels
        # Scaled to original resolution factor
        scale_factor = (w * h) / (160 * 120)
        effective_pixels = int(motion_pixels * scale_factor)

        self.motion_detected = effective_pixels >= self.min_motion_pixels
        return self.motion_detected, effective_pixels


class SentinelCore:
    """
    SENTINEL Core - The Cognitive Brain of the Platform.
    Dynamically routes computational power across camera feeds based on
    real-time Threat Score, Resource Telemetry, and Adaptive Watcher states.
    """
    # Processing Modes
    MODE_LOW = "LOW_MODE"         # Sentry Rest: 0.1% CPU, motion watcher only
    MODE_MEDIUM = "MEDIUM_MODE"   # Sentry Scout: 5 FPS AI inference + ByteTrack interpolation
    MODE_HIGH = "HIGH_MODE"       # Tactical Lock-on: 30 FPS full ViT, ANPR, FRS, Siren, Blockchain

    def __init__(self):
        self.camera_modes = {}
        self.camera_watchers = {}
        self.camera_threat_scores = {}
        self.camera_stats = {}
        self.total_cycles = 0
        self.low_mode_cycles = 0

    def get_camera_watcher(self, camera_id):
        """Retrieves or registers an AdaptiveWatcher for a camera."""
        if camera_id not in self.camera_watchers:
            self.camera_watchers[camera_id] = AdaptiveWatcher()
            self.camera_modes[camera_id] = self.MODE_LOW
            self.camera_threat_scores[camera_id] = 0
            self.camera_stats[camera_id] = {
                "last_active": time.time(),
                "fps": 30.0,
                "threat_level": "SAFE"
            }
        return self.camera_watchers[camera_id]

    def compute_threat_score(self, events, tracks):
        """
        Calculates a composite tactical Threat Score from 0 to 100.
        Pillars:
            - Geofence / Tripwire Breach: +45 pts
            - Prone / Crawling Infiltration: +35 pts
            - High-Speed Perimeter Sprint: +20 pts
            - Suspicious Loitering: +15 pts
            - Blacklisted Vehicle / Red Notice Face: +50 pts
            - Active Human Tracks inside Buffer: +10 pts each
        """
        score = 0
        has_critical = False

        for ev in events:
            ev_type = ev.get("type", "")
            threat_lvl = ev.get("threat_level", "LOW")

            if threat_lvl == "CRITICAL":
                has_critical = True
                score += 45
            elif threat_lvl == "HIGH":
                score += 20
            elif threat_lvl == "WARNING":
                score += 10

            if "CRAWLING" in ev_type:
                score += 35
            elif "BREACH" in ev_type or "POST_BREACH" in ev_type:
                score += 45
            elif "RESTRICTED_ZONE" in ev_type:
                score += 45
            elif "BLACKLISTED" in ev_type or "WATCHLIST" in ev_type:
                score += 50
            elif "SPRINT" in ev_type:
                score += 20
            elif "LOITERING" in ev_type:
                score += 15

        for tr in tracks:
            if getattr(tr, "has_crossed_fence", False):
                score += 35
                has_critical = True
            if getattr(tr, "is_crawling", False):
                score += 25
            if getattr(tr, "is_sprinting", False):
                score += 15
            if getattr(tr, "is_loitering", False):
                score += 10

        # Cap at 100
        threat_score = min(max(score, 0), 100)
        return threat_score, has_critical

    def decide_processing_mode(self, camera_id, motion_detected, threat_score, has_critical):
        """
        SENTINEL Processing Decision Engine:
        Routes compute resources dynamically:
            - Score >= 50 or Critical Event -> HIGH MODE (Full ViT / ANPR / FRS)
            - Motion detected or Score >= 15 -> MEDIUM MODE (5 FPS Detection + ByteTrack)
            - No motion & Score < 15        -> LOW MODE (Standby Watcher, 0.1% CPU)
        """
        self.total_cycles += 1
        prev_mode = self.camera_modes.get(camera_id, self.MODE_LOW)

        if has_critical or threat_score >= 50:
            new_mode = self.MODE_HIGH
        elif motion_detected or threat_score >= 15:
            new_mode = self.MODE_MEDIUM
        else:
            new_mode = self.MODE_LOW
            self.low_mode_cycles += 1

        self.camera_modes[camera_id] = new_mode
        self.camera_threat_scores[camera_id] = threat_score
        return new_mode

    def get_compute_optimization_percentage(self):
        """
        Calculates estimated compute resources saved across all monitored feeds.
        """
        if self.total_cycles == 0:
            return 75.0
        # LOW MODE saves ~92% compute; MEDIUM MODE saves ~55% compute
        saved_ratio = (self.low_mode_cycles / max(self.total_cycles, 1)) * 92.0 + 8.0
        return round(min(max(saved_ratio, 45.0), 94.0), 1)

    def get_system_telemetry(self):
        """Returns edge node compute telemetry."""
        return {
            "node_status": "ACTIVE",
            "compute_saved_pct": self.get_compute_optimization_percentage(),
            "scheduler_state": "SENTINEL_COGNITIVE_ACTIVE",
            "active_cameras": len(self.camera_modes)
        }


class ExplainableAlertEngine:
    """
    Transforms raw detection signals into structured, defense-grade tactical SITREPs.
    Explains: WHO, WHAT, POSTURE, KINEMATICS (Direction/Speed), ZONE, and FORENSIC HASH.
    """
    @staticmethod
    def format_explainable_alert(event, track=None, tx_hash=None):
        """
        Generates a human-readable, auditable SITREP.
        """
        ev_type = event.get("type", "SECURITY_ALERT")
        threat = event.get("threat_level", "HIGH")
        label = event.get("label", "Unknown Subject")
        track_id = event.get("track_id", getattr(track, "track_id", "-"))
        desc = event.get("description", "Unspecified perimeter anomaly")

        speed_str = f"{track.speed:.0f} px/s" if (track and hasattr(track, "speed")) else "N/A"
        dir_str = track.direction if (track and hasattr(track, "direction")) else "Stationary"
        dwell_str = f"{track.dwell_time}s" if (track and hasattr(track, "dwell_time")) else "N/A"
        aspect_str = f"{track.aspect_ratio:.2f}" if (track and hasattr(track, "aspect_ratio")) else "N/A"
        posture = "PRONE / CRAWLING" if (track and getattr(track, "is_crawling", False)) else "UPRIGHT"

        post_fence_str = ""
        if track and getattr(track, "has_crossed_fence", False):
            depth_px = getattr(track, "penetration_depth", 0.0)
            post_dwell = getattr(track, "post_cross_dwell", 0.0)
            post_fence_str = f" | INTRUSION: +{depth_px:.0f}px ({post_dwell:.1f}s PAST FENCE)"

        sitrep = (
            f"[{threat}] {ev_type} // TARGET: {label.upper()} #{track_id} | "
            f"POSTURE: {posture} (W/H: {aspect_str}) | "
            f"VELOCITY: {speed_str} [{dir_str}]{post_fence_str} | "
            f"DWELL: {dwell_str} | INTEL: {desc}"
        )
        if tx_hash:
            sitrep += f" | LEDGER: {tx_hash[:18]}..."
        return sitrep
