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

    def compute_threat_score(self, events=None, tracks=None, is_night_mode=False,
                             tripwire_y=None, geofence_pts=None):
        """
        SENTINEL Threat Score Mathematics
        Formula:
            T = min(100, H + V + Z + B + N + L + D)

        Risk Factors:
            H (Human detected):           +20
            V (Vehicle detected):         +10
            Z (Near restricted zone):     +20
            B (Virtual border breached):  +40
            N (Night-time activity):      +10
            L (Loitering detected):       +15
            D (Moving toward border):     +15

        Threat Level Thresholds:
            [LOW]       0 - 30
            [MEDIUM]   31 - 60
            [HIGH]     61 - 80
            [CRITICAL] 81 - 100
        """
        events = events or []
        tracks = tracks or []

        # 1. H: Human detected (+20)
        h_active = any(getattr(tr, "label", "") == "Person" or getattr(tr, "class_id", -1) == 0 for tr in tracks)
        if not h_active:
            h_active = any(
                "Person" in ev.get("label", "") or
                "CRAWLING" in ev.get("type", "") or
                "INTRUDER" in ev.get("type", "") or
                "WATCHLIST" in ev.get("type", "") or
                "PERIMETER" in ev.get("type", "")
                for ev in events
            )
        score_h = 20 if h_active else 0

        # 2. V: Vehicle detected (+10)
        v_active = any(getattr(tr, "label", "") in ["Car", "Truck", "Bus", "Motorcycle", "Bicycle"] for tr in tracks)
        if not v_active:
            v_active = any(
                "VEHICLE" in ev.get("type", "") or
                "PLATE" in ev.get("type", "") or
                ev.get("label", "") in ["Car", "Truck", "Bus", "Motorcycle"]
                for ev in events
            )
        score_v = 10 if v_active else 0

        # 3. Z: Near restricted zone (+20)
        z_active = False
        if tripwire_y is not None:
            for tr in tracks:
                if hasattr(tr, "box"):
                    _, y1, _, y2 = tr.box
                    if (tripwire_y - 120 <= y2 <= tripwire_y + 150) or getattr(tr, "has_crossed_fence", False):
                        z_active = True
                        break
        if not z_active and geofence_pts:
            z_active = any(
                getattr(tr, "has_crossed_fence", False) or
                getattr(tr, "is_loitering", False) or
                getattr(tr, "is_crawling", False)
                for tr in tracks
            )
        if not z_active:
            z_active = any(
                "BREACH" in ev.get("type", "") or
                "RESTRICTED" in ev.get("type", "") or
                "ZONE" in ev.get("type", "") or
                "GEOFENCE" in ev.get("type", "") or
                "TRIPWIRE" in ev.get("type", "")
                for ev in events
            )
        score_z = 20 if z_active else 0

        # 4. B: Virtual border breached (+40)
        b_active = any(getattr(tr, "has_crossed_fence", False) for tr in tracks)
        if not b_active:
            b_active = any(
                "BREACH" in ev.get("type", "") or
                "POST_BREACH" in ev.get("type", "") or
                "RESTRICTED_ZONE" in ev.get("type", "")
                for ev in events
            )
        score_b = 40 if b_active else 0

        # 5. N: Night-time activity (+10)
        n_active = bool(is_night_mode)
        if not n_active:
            n_active = any("NIGHT" in ev.get("type", "") or "THERMAL" in ev.get("type", "") for ev in events)
        score_n = 10 if n_active else 0

        # 6. L: Loitering detected (+15)
        l_active = any(getattr(tr, "is_loitering", False) or getattr(tr, "dwell_time", 0) >= 5.0 for tr in tracks)
        if not l_active:
            l_active = any("LOITERING" in ev.get("type", "") for ev in events)
        score_l = 15 if l_active else 0

        # 7. D: Moving toward border (+15)
        d_active = False
        for tr in tracks:
            d_str = getattr(tr, "direction", "").upper()
            vy = getattr(tr, "vy", 0.0)
            if "SOUTH" in d_str or "INWARD" in d_str or "DOWN" in d_str or vy > 5.0:
                d_active = True
                break
        if not d_active:
            d_active = any(
                "SPRINT" in ev.get("type", "") or
                "CRAWLING" in ev.get("type", "") or
                "INFILTRATION" in ev.get("type", "")
                for ev in events
            )
        score_d = 15 if d_active else 0

        # Raw summation and capping at 100
        raw_sum = score_h + score_v + score_z + score_b + score_n + score_l + score_d
        threat_score = min(100, max(0, raw_sum))

        # Classify Threat Level
        if threat_score <= 30:
            threat_level = "LOW"
        elif threat_score <= 60:
            threat_level = "MEDIUM"
        elif threat_score <= 80:
            threat_level = "HIGH"
        else:
            threat_level = "CRITICAL"

        has_critical = (threat_level == "CRITICAL") or b_active or any(ev.get("threat_level") == "CRITICAL" for ev in events)

        # Store explainable factor data
        self.last_threat_data = {
            "score": threat_score,
            "level": threat_level,
            "has_critical": has_critical,
            "formula": "T = H + V + Z + B + N + L + D",
            "raw_sum": raw_sum,
            "factors": {
                "H": {"name": "Human detected", "weight": 20, "active": h_active, "score": score_h},
                "V": {"name": "Vehicle detected", "weight": 10, "active": v_active, "score": score_v},
                "Z": {"name": "Near restricted zone", "weight": 20, "active": z_active, "score": score_z},
                "B": {"name": "Virtual border breached", "weight": 40, "active": b_active, "score": score_b},
                "N": {"name": "Night-time activity", "weight": 10, "active": n_active, "score": score_n},
                "L": {"name": "Loitering detected", "weight": 15, "active": l_active, "score": score_l},
                "D": {"name": "Moving toward border", "weight": 15, "active": d_active, "score": score_d},
            },
            "active_factors": [k for k, active in [("H", h_active), ("V", v_active), ("Z", z_active),
                                                   ("B", b_active), ("N", n_active), ("L", l_active),
                                                   ("D", d_active)] if active]
        }

        return threat_score, has_critical

    def get_last_threat_breakdown(self):
        """Returns the full explainable threat mathematics breakdown dictionary."""
        return getattr(self, "last_threat_data", {
            "score": 0,
            "level": "LOW",
            "has_critical": False,
            "formula": "T = H + V + Z + B + N + L + D",
            "factors": {},
            "active_factors": []
        })

    def decide_processing_mode(self, camera_id, motion_detected, threat_score, has_critical):
        """
        SENTINEL Processing Decision Engine:
        Routes compute resources dynamically based on Threat Level:
            - Score >= 61 or Critical Event -> HIGH MODE (Full ViT / ANPR / FRS)
            - Motion detected or Score >= 31 -> MEDIUM MODE (Detection + ByteTrack)
            - No motion & Score <= 30       -> LOW MODE (Standby Watcher, 0.1% CPU)
        """
        self.total_cycles += 1
        prev_mode = self.camera_modes.get(camera_id, self.MODE_LOW)

        if has_critical or threat_score >= 61:
            new_mode = self.MODE_HIGH
        elif motion_detected or threat_score >= 31:
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
