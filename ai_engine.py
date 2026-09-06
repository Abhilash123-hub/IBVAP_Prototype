"""
AI Vision Orchestration Engine (IBVAP)
Integrated with SENTINEL Cognitive Brain & ByteTrack Multi-Object Tracking.
Coordinates RT-DETR / YOLO Detection, ByteTrack Kinematic Tracking,
Adaptive Multi-Tier Compute Scheduling (LOW / MEDIUM / HIGH),
Behavioral Geofencing, Night Vision, and Explainable Defense Alerts.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO, RTDETR

from tracker import BorderObjectTracker, ByteTrackTracker
from face_engine import FaceRecognitionEngine
from behavior_engine import BehaviorEngine
from night_vision import NightVisionEngine
from sentinel_core import SentinelCore, ExplainableAlertEngine


class AIEngine:
    def __init__(self, model_path='rtdetr-l.pt'):
        print(f"Loading IBVAP Detection Model ({model_path})...")
        self.model_path = model_path
        try:
            if 'rtdetr' in model_path.lower():
                self.model = RTDETR(model_path)
                self.model_name = "RT-DETR-L (VISION TRANSFORMER)"
            else:
                self.model = YOLO(model_path)
                self.model_name = "YOLOv8n (BORDER CNN)"
        except Exception as e:
            print(f"Notice: Unable to initialize {model_path} ({e}). Falling back to local yolov8n.pt...")
            self.model = YOLO('yolov8n.pt')
            self.model_name = "YOLOv8n (BORDER CNN)"

        # COCO Class mapping:
        # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
        self.target_classes = {
            0: "Person",
            1: "Bicycle",
            2: "Car",
            3: "Motorcycle",
            5: "Bus",
            7: "Truck"
        }

        # Sub-engines
        self.tracker = ByteTrackTracker()
        self.face_engine = FaceRecognitionEngine()
        self.behavior_engine = BehaviorEngine()
        self.night_engine = NightVisionEngine()
        self.sentinel = SentinelCore()
        self.explainable_engine = ExplainableAlertEngine()

    def process_frame(self, frame, conf_threshold=0.35, use_clahe=False, use_thermal=False,
                      thermal_palette="inferno", geofence_pts=None, tripwire_y=None, enable_frs=True,
                      camera_id="CAM-01", force_mode=None):
        """
        Adaptive SENTINEL-guided processing of a surveillance frame.
        Dynamically throttles compute into LOW / MEDIUM / HIGH processing modes.
        Returns:
            processed_frame: Annotated tactical visualization
            tracks: list of active TrackedObject instances
            events: list of triggered security alerts
            faces: list of detected face matches
        """
        if frame is None:
            return None, [], [], []

        input_frame = frame.copy()

        # --- STEP 1: SENTINEL ADAPTIVE WATCHER MOTION CHECK ---
        watcher = self.sentinel.get_camera_watcher(camera_id)
        motion_detected, motion_pixels = watcher.check_motion(input_frame)

        # Preliminary threat check from active tracks
        active_tracks = list(self.tracker.tracks.values())
        threat_score, has_critical = self.sentinel.compute_threat_score([], active_tracks)

        # Determine Processing Mode
        if force_mode:
            current_mode = force_mode
            self.sentinel.camera_modes[camera_id] = current_mode
        else:
            current_mode = self.sentinel.decide_processing_mode(camera_id, motion_detected, threat_score, has_critical)

        # --- STEP 2: LOW MODE EXECUTION (Sentry Standby - Zero Heavy Inference) ---
        # If camera is in LOW MODE and no active tracks exist, return lightweight sentry frame
        if current_mode == SentinelCore.MODE_LOW and not active_tracks:
            vis_low = input_frame.copy()
            h_l, w_l = vis_low.shape[:2]
            # Draw subtle sentry scanline overlay
            cv2.line(vis_low, (0, 32), (w_l, 32), (0, 217, 255), 1)
            cv2.rectangle(vis_low, (0, 0), (w_l, 28), (7, 10, 15), -1)
            cv2.putText(vis_low, f"SENTINEL SENTRY // LOW MODE [IDLE - 0.1% CPU] // SENSOR: {camera_id}", (16, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 217, 255), 1)
            cv2.putText(vis_low, f"OPT: {self.sentinel.get_compute_optimization_percentage()}% SAVED", (w_l - 180, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (34, 197, 94), 1)
            return vis_low, [], [], []

        # --- STEP 3: OPTICAL ENHANCEMENTS (MEDIUM / HIGH MODE) ---
        if use_thermal:
            input_frame = self.night_engine.apply_thermal(input_frame, palette=thermal_palette)
        elif use_clahe:
            input_frame = self.night_engine.enhance_clahe(input_frame)

        # --- STEP 4: AI DETECTION (RT-DETR / YOLO) ---
        # In MEDIUM MODE, confidence threshold is slightly higher to prevent noise; in HIGH MODE, standard sensitivity
        effective_conf = conf_threshold if current_mode == SentinelCore.MODE_HIGH else max(conf_threshold, 0.35)
        results = self.model(input_frame, verbose=False)
        detections = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id in self.target_classes and conf >= 0.15:  # Allow low conf for ByteTrack secondary stage
                xyxy = box.xyxy[0].tolist()
                label = self.target_classes[cls_id]
                detections.append({
                    "class_id": cls_id,
                    "label": label,
                    "box": xyxy,
                    "confidence": conf
                })

        # --- STEP 5: BYTETRACK MULTI-OBJECT TRACKING ---
        active_tracks = self.tracker.update(detections)

        # --- STEP 6: BEHAVIORAL & VIRTUAL FENCE ANALYTICS ---
        events = []
        breached_tracks = set()

        # Polygon Geofence Check
        if geofence_pts and len(geofence_pts) >= 3:
            geo_intrusions = self.behavior_engine.check_polygon_intrusion(active_tracks, geofence_pts)
            for tr in geo_intrusions:
                breached_tracks.add(tr.track_id)
                ev = {
                    "type": "PERIMETER_GEOFENCE_BREACH",
                    "threat_level": "CRITICAL",
                    "track_id": tr.track_id,
                    "label": tr.label,
                    "description": f"Unauthorized {tr.label} breached Polygon Geofence Restricted Zone",
                    "box": tr.box
                }
                ev["explainable_sitrep"] = self.explainable_engine.format_explainable_alert(ev, tr)
                events.append(ev)

        # Tripwire Line Check
        if tripwire_y is not None:
            trip_intrusions = self.behavior_engine.check_tripwire(active_tracks, tripwire_y)
            for tr in trip_intrusions:
                breached_tracks.add(tr.track_id)
                ev = {
                    "type": "TRIPWIRE_BREACH",
                    "threat_level": "CRITICAL",
                    "track_id": tr.track_id,
                    "label": tr.label,
                    "description": f"Border tripwire breached by {tr.label} (ID: #{tr.track_id})",
                    "box": tr.box
                }
                ev["explainable_sitrep"] = self.explainable_engine.format_explainable_alert(ev, tr)
                events.append(ev)

        # Post-Fence Crossing Tracking (Monitors subjects across whole CCTV visual range after crossing)
        fence_ref_y = tripwire_y if tripwire_y is not None else (geofence_pts[0][1] if geofence_pts else None)
        post_fence_events = self.behavior_engine.check_post_fence_infiltration(active_tracks, fence_boundary_y=fence_ref_y, geofence_pts=geofence_pts)
        for pfe in post_fence_events:
            breached_tracks.add(pfe["track_id"])
            matched_tr = next((t for t in active_tracks if t.track_id == pfe["track_id"]), None)
            pfe["explainable_sitrep"] = self.explainable_engine.format_explainable_alert(pfe, matched_tr)
            events.append(pfe)

        # Persistent breach retention
        for tr in active_tracks:
            if getattr(tr, "has_crossed_fence", False):
                breached_tracks.add(tr.track_id)

        # Behavioral Anomalies (Crawling, Loitering, Sprinting)
        behavior_events = self.behavior_engine.analyze_behaviors(active_tracks)
        for bev in behavior_events:
            matched_tr = next((t for t in active_tracks if t.track_id == bev.get("track_id")), None)
            bev["explainable_sitrep"] = self.explainable_engine.format_explainable_alert(bev, matched_tr)
            events.append(bev)

        # Recalculate dynamic Threat Score with detected events
        final_threat_score, is_critical = self.sentinel.compute_threat_score(events, active_tracks)
        # Escalate to HIGH MODE if critical event detected
        if is_critical or final_threat_score >= 50:
            current_mode = SentinelCore.MODE_HIGH
            self.sentinel.camera_modes[camera_id] = current_mode

        # --- STEP 7: FACIAL RECOGNITION SYSTEM (HIGH MODE OR EXPLICIT ENABLE) ---
        face_matches = []
        if enable_frs and (current_mode == SentinelCore.MODE_HIGH or len(active_tracks) > 0):
            human_boxes = [tr.box for tr in active_tracks if tr.label == "Person"]
            if human_boxes:
                detected_faces = self.face_engine.detect_faces(frame, human_boxes)
                for fbox in detected_faces:
                    fx1, fy1, fx2, fy2 = map(int, fbox)
                    face_crop = frame[max(0, fy1):min(fy2, frame.shape[0]), max(0, fx1):min(fx2, frame.shape[1])]
                    if face_crop.size > 100:
                        match = self.face_engine.match_face(face_crop)
                        match["box"] = fbox
                        face_matches.append(match)

                        if match["matched"] and match["threat"] in ["CRITICAL", "HIGH"]:
                            f_ev = {
                                "type": "WATCHLIST_FACE_IDENTIFIED",
                                "threat_level": match["threat"],
                                "track_id": 0,
                                "label": "Person",
                                "description": f"Wanted Subject Identified: {match['name']} ({match['role']})",
                                "box": fbox,
                                "face_name": match["name"]
                            }
                            f_ev["explainable_sitrep"] = self.explainable_engine.format_explainable_alert(f_ev)
                            events.append(f_ev)

        # --- STEP 8: RENDER TACTICAL C2 HUD ---
        display_frame = self.render_tactical_hud(
            frame=input_frame if (use_thermal or use_clahe) else frame,
            tracks=active_tracks,
            breached_track_ids=breached_tracks,
            geofence_pts=geofence_pts,
            tripwire_y=tripwire_y,
            face_matches=face_matches,
            camera_id=camera_id,
            mode=current_mode,
            threat_score=final_threat_score
        )

        return display_frame, active_tracks, events, face_matches

    def render_tactical_hud(self, frame, tracks, breached_track_ids, geofence_pts=None,
                            tripwire_y=None, face_matches=None, camera_id="CAM-01",
                            mode=SentinelCore.MODE_MEDIUM, threat_score=0):
        """Draws professional military/C2 tactical overlays onto the video frame."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # 1. Top HUD Header with SENTINEL Brain Mode & Threat Meter
        cv2.rectangle(vis, (0, 0), (w, 30), (7, 10, 15), -1)
        mode_color = (0, 0, 255) if mode == SentinelCore.MODE_HIGH else ((0, 217, 255) if mode == SentinelCore.MODE_MEDIUM else (34, 197, 94))
        mode_tag = f"SENTINEL: [{mode.replace('_', ' ')}] // SCORE: {threat_score}/100"
        cv2.putText(vis, mode_tag, (16, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, mode_color, 1)

        opt_text = f"COMPUTE SAVED: {self.sentinel.get_compute_optimization_percentage()}% | TRACKER: BYTETRACK"
        cv2.putText(vis, opt_text, (w - 380, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 217, 255), 1)

        # Visual Range Zone Labels (Separating pre-fence approach and post-fence domestic territory)
        cv2.putText(vis, "SECTOR-07: BUFFER / APPROACH SECTOR", (w - 295, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (139, 148, 158), 1)
        cv2.putText(vis, "RESTRICTED DOMESTIC TERRITORY // SENSOR RANGE", (w - 365, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (139, 148, 158), 1)

        # 2. Draw Geofence Polygon (Calibrated narrow barrier strip)
        is_geo_breached = any(tid in breached_track_ids for tid in [t.track_id for t in tracks])
        if geofence_pts and len(geofence_pts) >= 3:
            vis = self.behavior_engine.draw_geofence(vis, geofence_pts, is_breached=is_geo_breached)

        # 3. Draw Tripwire Line
        if tripwire_y is not None:
            vis = self.behavior_engine.draw_tripwire(vis, tripwire_y, is_breached=is_geo_breached)

        # 4. Draw ByteTrack Path Trajectories & Tactical Corner Reticles
        for tr in tracks:
            x1, y1, x2, y2 = map(int, tr.box)
            has_crossed = getattr(tr, "has_crossed_fence", False)
            depth_px = getattr(tr, "penetration_depth", 0.0)
            post_dwell = getattr(tr, "post_cross_dwell", 0.0)
            is_threat = (has_crossed or tr.track_id in breached_track_ids or tr.is_crawling or tr.is_sprinting or tr.is_loitering)

            # Draw ByteTrack historical path polyline
            if len(tr.path) > 1:
                trail_color = (0, 0, 255) if is_threat else (0, 217, 255)
                for i in range(1, len(tr.path)):
                    thickness = int(np.sqrt(float(i + 1)) * 1.1)
                    cv2.line(vis, tr.path[i - 1], tr.path[i], trail_color, max(thickness, 1))

            # Color scheme & Tactical Status Tag
            if has_crossed:
                box_color = (0, 0, 255)      # Red Alert
                status_tag = f"INTRUDER [POST-FENCE: +{depth_px:.0f}px | {post_dwell:.0f}s]"
            elif tr.track_id in breached_track_ids:
                box_color = (0, 0, 255)      # Red
                status_tag = "INTRUDER [BREACH]"
            elif tr.is_crawling:
                box_color = (0, 50, 255)     # Deep Orange / Red
                status_tag = "PRONE / CRAWL"
            elif tr.is_loitering:
                box_color = (0, 165, 255)    # Amber
                status_tag = f"LOITERING ({tr.duration}s)"
            elif tr.is_sprinting:
                box_color = (0, 200, 255)    # Orange
                status_tag = "SPRINT / EVASION"
            elif tr.label == "Person":
                box_color = (0, 255, 0)      # Green
                status_tag = "HUMAN TARGET"
            else:
                box_color = (255, 180, 0)    # Cyan / Blue for vehicles
                status_tag = f"{tr.label.upper()}"

            # If post-fence intruder, draw inward penetration vector indicator
            if has_crossed and tripwire_y is not None and y2 > tripwire_y:
                cx = int((x1 + x2) / 2)
                cv2.arrowedLine(vis, (cx, tripwire_y), (cx, int((y1 + y2) / 2)), (0, 0, 255), 2, tipLength=0.25)

            # Tactical bounding box (Corner brackets style)
            line_len = min(int((x2 - x1) * 0.25), 25)
            thick = 2
            cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 1)
            # Brackets
            cv2.line(vis, (x1, y1), (x1 + line_len, y1), box_color, thick + 1)
            cv2.line(vis, (x1, y1), (x1, y1 + line_len), box_color, thick + 1)
            cv2.line(vis, (x2, y1), (x2 - line_len, y1), box_color, thick + 1)
            cv2.line(vis, (x2, y1), (x2, y1 + line_len), box_color, thick + 1)
            cv2.line(vis, (x1, y2), (x1 + line_len, y2), box_color, thick + 1)
            cv2.line(vis, (x1, y2), (x1, y2 - line_len), box_color, thick + 1)
            cv2.line(vis, (x2, y2), (x2 - line_len, y2), box_color, thick + 1)
            cv2.line(vis, (x2, y2), (x2, y2 - line_len), box_color, thick + 1)

            # Target Telemetry Badge: ID, Status, Confidence
            badge_text = f"ID:#{tr.track_id} {status_tag} | {tr.confidence:.2f}"
            t_size = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0]
            cv2.rectangle(vis, (x1, max(0, y1 - 20)), (x1 + t_size[0] + 8, y1), box_color, -1)
            cv2.putText(vis, badge_text, (x1 + 4, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)

            # ByteTrack Kinematics Tag: Direction, Speed, Duration
            kinematic_text = f"[{tr.direction}] {tr.speed:.0f} px/s | {tr.duration}s"
            cv2.putText(vis, kinematic_text, (x1, min(h - 5, y2 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, box_color, 1)

        # 5. Draw Face Match Indicators
        if face_matches:
            for f in face_matches:
                fx1, fy1, fx2, fy2 = map(int, f["box"])
                f_color = (0, 0, 255) if f["threat"] in ["CRITICAL", "HIGH"] else (0, 255, 120)
                cv2.rectangle(vis, (fx1, fy1), (fx2, fy2), f_color, 2)
                f_tag = f"FRS: {f['name']} [{f['threat']}]"
                cv2.putText(vis, f_tag, (fx1, max(12, fy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, f_color, 1)

        return vis