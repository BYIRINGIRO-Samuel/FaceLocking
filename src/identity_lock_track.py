# src/identity_lock_track.py
"""
Identity Lock: recognizes and "locks onto" an enrolled/registered person.
- Enrolled Identity: locked (green box), shows name + live expression + blink count.
- Stranger: detected in red, not locked, expression analysis ignored.
- Servo motor movement is currently disabled (ENABLE_SERVO = False).

Run:
    python -m src.identity_lock_track
    python -m src.identity_lock_track --cam 1   (use laptop webcam)

Keys:
    q   : quit
    r   : reload face DB
    +/- : adjust distance threshold live
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Use HaarFaceMesh5pt from recognize (reliable ROI-based FaceMesh, identical to working recognize.py)
from .haar_5pt import align_face_5pt
from .recognize import ArcFaceEmbedderONNX, HaarFaceMesh5pt, load_db_npz, FaceDBMatcher
from .expression import ExpressionAnalyzer

# -------------------------
# Servo -- DISABLED for now
# -------------------------
ENABLE_SERVO = False  # flip to True + see track_and_follow.py to re-enable motor movement

# -------------------------
# Recognition tuning
# -------------------------
DB_PATH = Path("data/db/face_db.npz")
DEFAULT_DIST_THRESH = 0.40  # wider = more lenient, lower = stricter


def draw_label_with_outline(img: np.ndarray, text: str, pos: tuple[int, int], scale: float = 0.7,
                            color: tuple[int, int, int] = (0, 255, 0), thickness: int = 2):
    """Draw text with a black background outline so it's always readable."""
    x, y = pos
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Identity Lock & Face Tracking Demo")
    parser.add_argument("--cam", type=int, default=0, help="Camera index (default 0, try 1 for laptop cam)")
    parser.add_argument("--thresh", type=float, default=DEFAULT_DIST_THRESH, help="Distance threshold")
    args = parser.parse_args()

    # Open camera with DSHOW backend
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[identity_lock] Warning: Camera {args.cam} failed to open. Trying fallback camera...")
        fallback = 1 if args.cam == 0 else 0
        cap = cv2.VideoCapture(fallback, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {args.cam} or fallback {fallback}.")
        print(f"[identity_lock] Opened fallback camera {fallback}.")

    det = HaarFaceMesh5pt(min_size=(70, 70), debug=False)
    embedder = ArcFaceEmbedderONNX(model_path="models/embedder_arcface.onnx", input_size=(112, 112), debug=False)
    expr_analyzer = ExpressionAnalyzer(debug=False)

    db = load_db_npz(DB_PATH)
    if not db:
        raise RuntimeError(f"No enrolled identities found at {DB_PATH}. Run enroll.py first.")
    matcher = FaceDBMatcher(db=db, dist_thresh=args.thresh)
    print(f"[identity_lock] Loaded {len(matcher._names)} identities: {matcher._names}")
    print(f"[identity_lock] Distance threshold: {matcher.dist_thresh:.2f}")

    if ENABLE_SERVO:
        print("[identity_lock] ENABLE_SERVO is True (see track_and_follow.py for MQTT logic).")

    print("\nIdentity Lock running. Keys: 'q'=quit, 'r'=reload DB, '+/-'=tune threshold")

    t0 = time.time()
    frames = 0
    fps: float | None = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[identity_lock] Frame grab failed.")
                break

            H, W = frame.shape[:2]
            vis = frame.copy()

            # Measure FPS
            frames += 1
            dt = time.time() - t0
            if dt >= 1.0:
                fps = frames / dt
                frames = 0
                t0 = time.time()

            faces = det.detect(frame, max_faces=1)

            if faces:
                f = faces[0]
                aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                emb = embedder.embed(aligned)
                mr = matcher.match(emb)

                print(f"[identity_lock] name={mr.name} dist={mr.distance:.3f} accepted={mr.accepted}")

                # Ensure coordinates are cleanly bounded
                x1 = max(0, min(W - 1, f.x1))
                y1 = max(0, min(H - 1, f.y1))
                x2 = max(0, min(W - 1, f.x2))
                y2 = max(0, min(H - 1, f.y2))

                if mr.accepted:
                    # KNOWN person -- LOCKED
                    color = (0, 255, 0)
                    lock_label = f"LOCKED: {mr.name}"

                    # Safe ROI crop for expression analysis
                    face_roi = frame[y1:y2, x1:x2]
                    expr = expr_analyzer.analyze(face_roi) if (y2 - y1 > 20 and x2 - x1 > 20) else None

                    # Draw Bounding Box & 5 keypoints
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    for (kx, ky) in f.kps.astype(int):
                        cv2.circle(vis, (int(kx), int(ky)), 2, color, -1)

                    # Dynamic safe text placement (avoid clipping at frame boundaries)
                    if y1 >= 55:
                        y_name = y1 - 32
                        y_dist = y1 - 10
                        y_expr = min(H - 35, y2 + 22)
                        y_blink = min(H - 12, y2 + 45)
                    else:
                        y_name = min(H - 80, y2 + 22)
                        y_dist = min(H - 60, y2 + 42)
                        y_expr = min(H - 38, y2 + 64)
                        y_blink = min(H - 16, y2 + 86)

                    draw_label_with_outline(vis, lock_label, (x1, y_name), scale=0.75, color=color, thickness=2)
                    draw_label_with_outline(vis, f"dist={mr.distance:.3f}", (x1, y_dist), scale=0.55, color=(200, 255, 200), thickness=1)

                    if expr:
                        # Color code expression
                        if expr.expression == "Smiling":
                            expr_color = (0, 255, 255)   # Yellow
                        elif expr.expression == "Sad":
                            expr_color = (255, 100, 255) # Magenta/Pink
                        else:
                            expr_color = (220, 220, 220) # Off-white / Neutral

                        # Eye status color
                        eye_str = "CLOSED" if expr.eyes_closed else "OPEN"
                        eye_color = (0, 140, 255) if expr.eyes_closed else (0, 255, 128)

                        draw_label_with_outline(vis, f"Expr: {expr.expression}", (x1, y_expr), scale=0.65, color=expr_color, thickness=2)
                        draw_label_with_outline(vis, f"Eyes: {eye_str} | Blinks: {expr.blink_count}", (x1, y_blink), scale=0.65, color=eye_color, thickness=2)
                    else:
                        draw_label_with_outline(vis, "Expr: Analyzing...", (x1, y_expr), scale=0.6, color=(180, 180, 180), thickness=1)

                else:
                    # STRANGER -- detected but not locked
                    color = (0, 0, 255)
                    expr_analyzer.reset_blink_count()

                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    for (kx, ky) in f.kps.astype(int):
                        cv2.circle(vis, (int(kx), int(ky)), 2, color, -1)

                    y_text = y1 - 10 if y1 >= 30 else min(H - 15, y2 + 25)
                    draw_label_with_outline(vis, "Stranger", (x1, y_text), scale=0.75, color=color, thickness=2)
                    draw_label_with_outline(vis, f"dist={mr.distance:.3f}", (x1, y_text + 20 if y1 < 30 else y1 + 18),
                                            scale=0.55, color=(200, 200, 255), thickness=1)

            # Top Status Bar
            header_bg = np.zeros((35, W, 3), dtype=np.uint8)
            vis[0:35, 0:W] = cv2.addWeighted(vis[0:35, 0:W], 0.3, header_bg, 0.7, 0)

            fps_str = f"fps: {fps:.1f}" if fps is not None else "fps: --"
            header_text = f"Identity Lock | IDs: {len(matcher._names)} | thr={matcher.dist_thresh:.2f} | {fps_str}"
            cv2.putText(vis, header_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Identity Lock & Track", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                matcher.reload_from(DB_PATH)
                print(f"[identity_lock] Reloaded DB: {len(matcher._names)} identities")
            elif key in (ord("+"), ord("=")):
                matcher.dist_thresh = float(min(1.20, matcher.dist_thresh + 0.02))
                print(f"[identity_lock] Threshold increased: {matcher.dist_thresh:.2f}")
            elif key == ord("-"):
                matcher.dist_thresh = float(max(0.05, matcher.dist_thresh - 0.02))
                print(f"[identity_lock] Threshold decreased: {matcher.dist_thresh:.2f}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()