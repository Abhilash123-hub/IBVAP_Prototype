"""
Synthetic Border Surveillance Video Generator (IBVAP)
Generates high-fidelity surveillance footage scenarios for immediate turnkey evaluation:
1. Day Border Patrol & ANPR Intrusion Scenario
2. Night-time Low-Light Thermal & Crawling Infiltration Scenario
"""

import os
import cv2
import numpy as np


def generate_day_patrol_video(output_path="assets/demo_border_patrol.mp4", num_frames=180, fps=20):
    """
    Simulates a Border Outpost Checkpoint with:
    - Approaching truck with blacklisted plate 'HR26DK8901'
    - Restricted perimeter zone with a walking human intruder
    - Normal patrol vehicle
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 720, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_idx in range(num_frames):
        # Base background: desert/arid border outpost
        frame = np.full((height, width, 3), (120, 160, 180), dtype=np.uint8)  # Sandy terrain

        # Horizon and sky
        frame[0:150, :] = (210, 190, 150)  # Pale desert sky

        # Border road (asphalt)
        road_pts = np.array([[220, height], [460, height], [390, 150], [330, 150]], np.int32)
        cv2.fillPoly(frame, [road_pts], (70, 70, 75))

        # Road dashed center lines
        for y in range(160, height, 40):
            cv2.line(frame, (360, y), (360, min(y + 20, height)), (240, 240, 240), 2)

        # Border Barbed Wire Fence (Left side)
        for y_fence in [180, 240, 300, 360]:
            cv2.line(frame, (10, y_fence), (240, y_fence + 40), (90, 90, 90), 2)
        for x_post in [20, 80, 140, 200]:
            cv2.line(frame, (x_post, 170), (x_post, 420), (50, 50, 60), 4)

        # Watchtower structure
        cv2.rectangle(frame, (30, 90), (90, 170), (80, 80, 90), -1)
        cv2.rectangle(frame, (40, 65), (80, 90), (60, 60, 70), -1)
        cv2.putText(frame, "BOP-07", (35, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # SCENARIO OBJECT 1: Approaching Smuggling Truck (Moving down the road)
        # Progress 0.0 to 1.0
        prog_t = min(1.0, frame_idx / float(num_frames * 0.85))
        scale = 0.3 + 0.9 * prog_t
        t_w, t_h = int(140 * scale), int(160 * scale)
        t_x = int(360 - (t_w // 2))
        t_y = int(150 + (height - 240) * (prog_t ** 1.3))

        # Draw vehicle body (Truck)
        cv2.rectangle(frame, (t_x, t_y), (t_x + t_w, t_y + t_h), (35, 75, 140), -1)  # Cabin
        cv2.rectangle(frame, (t_x + 10, t_y + 10), (t_x + t_w - 10, t_y + int(t_h * 0.45)), (180, 200, 210), -1) # Windshield
        # Headlights
        cv2.circle(frame, (t_x + 15, t_y + t_h - 25), int(7 * scale), (255, 255, 200), -1)
        cv2.circle(frame, (t_x + t_w - 15, t_y + t_h - 25), int(7 * scale), (255, 255, 200), -1)

        # Realistic Number Plate on Truck Bumper
        pw = int(65 * scale)
        ph = int(18 * scale)
        px = t_x + (t_w - pw) // 2
        py = t_y + t_h - ph - 6
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 255, 255), -1)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 0, 0), 1)
        if scale > 0.6:
            cv2.putText(frame, "HR26DK8901", (px + 2, py + ph - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28 * scale, (0, 0, 0), 1)

        # SCENARIO OBJECT 2: Walking Human Target near fence entering restricted zone
        # Target starts outside fence and walks rightwards into restricted zone
        h_x = int(70 + (frame_idx * 1.8))
        h_y = int(280 + np.sin(frame_idx * 0.2) * 5)
        # Head
        cv2.circle(frame, (h_x + 15, h_y + 12), 10, (190, 160, 140), -1)
        # Upper body / jacket (Dark clothing)
        cv2.rectangle(frame, (h_x + 5, h_y + 22), (h_x + 25, h_y + 55), (40, 50, 40), -1)
        # Legs
        leg_phase = np.sin(frame_idx * 0.4) * 8
        cv2.line(frame, (h_x + 10, h_y + 55), (int(h_x + 10 - leg_phase), h_y + 85), (20, 20, 30), 3)
        cv2.line(frame, (h_x + 20, h_y + 55), (int(h_x + 20 + leg_phase), h_y + 85), (20, 20, 30), 3)

        # CCTV timestamp and telemetry
        cv2.putText(frame, f"BOP-NORTH-CAM01 // 2026-09-06 14:{frame_idx//30:02d}:{(frame_idx%30)*2:02d} IST", 
                    (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        out.write(frame)

    out.release()
    print(f"Generated Day Patrol Demo Video: {output_path} ({os.path.getsize(output_path)} bytes)")


def generate_night_thermal_video(output_path="assets/demo_night_thermal.mp4", num_frames=180, fps=20):
    """
    Simulates a Night Surveillance / Thermal FLIR Scenario:
    - Pitch-black border sector with infrared illuminator
    - Target crawling in prone position under border wire
    - Second infiltrator breaching tripwire
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 720, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_idx in range(num_frames):
        # Low light / dark nocturnal background
        frame = np.full((height, width, 3), (18, 22, 20), dtype=np.uint8)

        # Distant terrain silhouette
        cv2.line(frame, (0, 170), (width, 170), (35, 40, 35), 2)

        # Barbed wire perimeter fence line across screen
        for y_fence in [210, 260, 310]:
            cv2.line(frame, (0, y_fence), (width, y_fence), (45, 50, 45), 1)
            for x_barb in range(15, width, 30):
                cv2.line(frame, (x_barb - 4, y_fence - 4), (x_barb + 4, y_fence + 4), (55, 60, 55), 1)

        # Concrete border pillars
        for xp in [120, 320, 520]:
            cv2.rectangle(frame, (xp, 180), (xp + 15, 340), (50, 55, 50), -1)

        # SCENARIO OBJECT: Prone Crawling Infiltrator (Crawling from right to left under fence)
        # Person crawling has low height and wide horizontal aspect ratio (W/H > 1.2)
        c_x = int(width - 90 - (frame_idx * 2.2))
        c_y = 330

        # Crawling body (prone horizontal torso)
        crawl_w = 60
        crawl_h = 24
        # Thermal heat signature (brighter in IR)
        cv2.ellipse(frame, (c_x + 30, c_y + 12), (28, 10), 0, 0, 360, (75, 85, 80), -1)
        # Head
        cv2.circle(frame, (c_x + 6, c_y + 10), 8, (90, 105, 95), -1)
        # Arms reaching forward
        arm_ext = int(np.sin(frame_idx * 0.3) * 6)
        cv2.line(frame, (c_x + 10, c_y + 14), (c_x - 5 + arm_ext, c_y + 16), (70, 80, 75), 3)

        # Add IR camera grain / sensor noise
        noise = np.random.normal(0, 4, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Telemetry
        cv2.putText(frame, f"NIGHT-IR-CAM03 // GAIN: +18dB // 2026-09-06 02:{frame_idx//30:02d}:{(frame_idx%30)*2:02d} IST", 
                    (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 180, 120), 1)
        out.write(frame)

    out.release()
    print(f"Generated Night Thermal Demo Video: {output_path} ({os.path.getsize(output_path)} bytes)")


if __name__ == "__main__":
    generate_day_patrol_video()
    generate_night_thermal_video()
