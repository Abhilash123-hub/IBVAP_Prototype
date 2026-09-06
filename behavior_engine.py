"""
Behavioral Analytics & Virtual Fence Engine (IBVAP)
Detects geofence polygon intrusions, tripwire breaches, loitering, crawling/prone infiltration,
and high-speed border perimeter sprinting.
"""

import cv2
import numpy as np


class BehaviorEngine:
    def __init__(self, loiter_threshold_sec=5.0, sprint_speed_px=140.0, crawl_aspect_ratio=0.95):
        self.loiter_threshold_sec = loiter_threshold_sec
        self.sprint_speed_px = sprint_speed_px
        self.crawl_aspect_ratio = crawl_aspect_ratio

    def check_polygon_intrusion(self, tracks, polygon_points):
        """
        Checks which tracks are inside the polygon geofence.
        polygon_points: list of [x, y] coordinates forming a closed polygon.
        Returns list of intruding tracks and their status.
        """
        if polygon_points is None or len(polygon_points) < 3:
            return []

        pts = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
        intrusions = []

        for tr in tracks:
            # Check bottom center of bounding box (footprint) and center
            x1, y1, x2, y2 = tr.box
            foot_x = (x1 + x2) / 2.0
            foot_y = y2
            center_y = (y1 + y2) / 2.0

            foot_dist = cv2.pointPolygonTest(pts, (float(foot_x), float(foot_y)), False)
            center_dist = cv2.pointPolygonTest(pts, (float(foot_x), float(center_y)), False)

            # If inside or on edge (distance >= 0)
            if foot_dist >= 0 or center_dist >= 0:
                tr.mark_fence_breached(cross_point=(int(foot_x), int(foot_y)))
                intrusions.append(tr)

        return intrusions

    def check_tripwire(self, tracks, tripwire_y):
        """
        Checks if any target crosses the horizontal tripwire line.
        tripwire_y: vertical coordinate of horizontal line.
        """
        intrusions = []
        for tr in tracks:
            x1, y1, x2, y2 = tr.box
            foot_x = int((x1 + x2) / 2.0)
            foot_y = int(y2)

            # Check current position vs previous position
            if y2 >= tripwire_y and y1 <= tripwire_y + 40:
                tr.mark_fence_breached(cross_point=(foot_x, foot_y))
                intrusions.append(tr)
            elif len(tr.history) >= 2:
                prev_y = tr.history[-2][1]
                curr_y = tr.history[-1][1]
                if (prev_y < tripwire_y <= curr_y) or (prev_y > tripwire_y >= curr_y):
                    tr.mark_fence_breached(cross_point=(foot_x, foot_y))
                    intrusions.append(tr)
        return intrusions

    def check_post_fence_infiltration(self, tracks, fence_boundary_y=None, geofence_pts=None):
        """
        Monitors targets across the entire visual range of the CCTV after they cross the virtual fence.
        Calculates:
          - Penetration depth past the virtual fence into domestic territory.
          - Post-crossing dwell time (seconds elapsed inside protected zone).
          - Generates POST_BREACH_INFILTRATION events for active intruders.
        """
        post_breach_events = []
        for tr in tracks:
            if getattr(tr, "has_crossed_fence", False):
                # Update penetration telemetry with respect to fence line
                tr.update_post_breach_telemetry(fence_ref_y=fence_boundary_y)

                depth_px = getattr(tr, "penetration_depth", 0.0)
                post_dwell = getattr(tr, "post_cross_dwell", 0.0)
                depth_m = round(depth_px * 0.2, 1)  # Calibrated: 1 px approx 0.2 meters

                # Trigger post-breach tracking event if advancing or lingering inside territory
                post_breach_events.append({
                    "type": "POST_BREACH_INFILTRATION",
                    "threat_level": "CRITICAL",
                    "track_id": tr.track_id,
                    "label": tr.label,
                    "description": f"Intruder {tr.label} (ID: #{tr.track_id}) active inside restricted territory for {post_dwell:.1f}s after crossing virtual fence (Penetration: {depth_px:.0f}px / {depth_m}m, Speed: {tr.speed:.0f} px/s [{tr.direction}])",
                    "box": tr.box,
                    "penetration_depth": depth_px,
                    "penetration_depth_m": depth_m,
                    "post_dwell": post_dwell
                })
        return post_breach_events

    def analyze_behaviors(self, tracks, geofence_pts=None, tripwire_y=None):
        """
        Comprehensive behavioral scan:
        Returns list of suspicious events with threat assessments.
        """
        events = []

        for tr in tracks:
            # 1. Crawling / Prone Infiltration Detection (Only for humans)
            if tr.label == "Person":
                # When a person is prone/crawling, width becomes greater than or comparable to height
                if tr.aspect_ratio >= self.crawl_aspect_ratio:
                    tr.is_crawling = True
                    events.append({
                        "type": "CRAWLING_INFILTRATION",
                        "threat_level": "CRITICAL",
                        "track_id": tr.track_id,
                        "label": tr.label,
                        "description": f"Suspect in prone/crawling posture detected (W/H ratio: {tr.aspect_ratio:.2f})",
                        "box": tr.box
                    })
                else:
                    tr.is_crawling = False

                # 2. Sprinting / Rapid Evasion Detection
                if tr.speed > self.sprint_speed_px:
                    tr.is_sprinting = True
                    events.append({
                        "type": "SPRINT_EVASION",
                        "threat_level": "HIGH",
                        "track_id": tr.track_id,
                        "label": tr.label,
                        "description": f"Rapid sprint movement towards border ({tr.speed:.0f} px/s, {tr.direction})",
                        "box": tr.box
                    })
                else:
                    tr.is_sprinting = False

                # 3. Loitering Detection (Differentiates pre-breach approach vs post-breach territory loitering)
                if tr.dwell_time >= self.loiter_threshold_sec and tr.speed < 40:
                    tr.is_loitering = True
                    if getattr(tr, "has_crossed_fence", False):
                        post_dwell = getattr(tr, "post_cross_dwell", tr.dwell_time)
                        depth_px = getattr(tr, "penetration_depth", 0.0)
                        depth_m = round(depth_px * 0.2, 1)
                        events.append({
                            "type": "RESTRICTED_ZONE_LOITERING",
                            "threat_level": "CRITICAL",
                            "track_id": tr.track_id,
                            "label": tr.label,
                            "description": f"Intruder {tr.label} (ID: #{tr.track_id}) loitering INSIDE restricted territory for {post_dwell:.1f}s after crossing virtual fence (Penetration: {depth_px:.0f}px / {depth_m}m)",
                            "box": tr.box
                        })
                    else:
                        events.append({
                            "type": "SUSPICIOUS_LOITERING",
                            "threat_level": "HIGH",
                            "track_id": tr.track_id,
                            "label": tr.label,
                            "description": f"Target loitering near perimeter approach zone for {tr.dwell_time:.1f}s",
                            "box": tr.box
                        })
                else:
                    tr.is_loitering = False

        return events

    def draw_geofence(self, frame, polygon_points, is_breached=False):
        """Draws semi-transparent polygon zone with boundary glow."""
        if polygon_points is None or len(polygon_points) < 3:
            return frame

        overlay = frame.copy()
        pts = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
        fill_color = (0, 0, 180) if is_breached else (0, 140, 255)
        border_color = (0, 0, 255) if is_breached else (0, 180, 255)

        cv2.fillPoly(overlay, [pts], fill_color)
        alpha = 0.24
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw outer border with tactical glow
        cv2.polylines(frame, [pts], isClosed=True, color=border_color, thickness=2, lineType=cv2.LINE_AA)

        # Draw Zone label
        M = cv2.moments(pts)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            status_text = "[ALERT] VIRTUAL FENCE BREACHED" if is_breached else "VIRTUAL FENCE BARRIER // RESTRICTED STRIP"
            text_color = (0, 0, 255) if is_breached else (0, 220, 255)
            cv2.putText(frame, status_text, (cX - 140, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
            cv2.putText(frame, status_text, (cX - 140, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1)

        return frame

    def draw_tripwire(self, frame, tripwire_y, is_breached=False):
        """Draws horizontal virtual tripwire."""
        w = frame.shape[1]
        line_color = (0, 0, 255) if is_breached else (0, 69, 255)
        cv2.line(frame, (0, tripwire_y), (w, tripwire_y), line_color, 2, lineType=cv2.LINE_AA)
        
        # Tactical dashed tick marks
        for x in range(0, w, 40):
            cv2.line(frame, (x, tripwire_y - 4), (x, tripwire_y + 4), line_color, 1)

        label = "[BREACH] VIRTUAL FENCE LINE BREACHED" if is_breached else "VIRTUAL FENCE LINE // PERIMETER ZERO-LINE BARRIER"
        cv2.putText(frame, label, (20, tripwire_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
        cv2.putText(frame, label, (20, tripwire_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, line_color, 1)
        return frame
