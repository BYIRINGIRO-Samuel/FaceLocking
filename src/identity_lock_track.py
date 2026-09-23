# src/identity_lock_track.py
"""
Identity Lock: only recognizes and "locks onto" an enrolled/registered
person. Strangers are detected but ignored (not locked, no expression
shown). Servo motor movement is DISABLED for now (ENABLE_SERVO = False)
-- this version only does visual lock + expression/blink display, no
motor commands sent. Flip ENABLE_SERVO to True later to re-enable
tracking motion (see track_and_follow.py for that logic).

Run:
    python -m src.identity_lock_track

Keys:
    q : quit
"""
from pathlib import Path

import cv2

from .haar_5pt import Haar5ptDetector, align_face_5pt
from .recognize import ArcFaceEmbedderONNX, load_db_npz, FaceDBMatcher
from .expression import ExpressionAnalyzer

# -------------------------
# Servo -- DISABLED for now
# -------------------------
ENABLE_SERVO = False  # flip to True + see track_and_follow.py to re-enable motor movement

# -------------------------
# Recognition tuning
# -------------------------
DB_PATH = Path("data/db/face_db.npz")
DIST_THRESH = 0.40  # from your evaluate.py run -- adjust if needed, wider = more lenient


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Tracking camera (index 0) not opened.")

    det = Haar5ptDetector(min_size=(70, 70), smooth_alpha=0.80, debug=False)
    embedder = ArcFaceEmbedderONNX(model_path="models/embedder_arcface.onnx", input_size=(112, 112), debug=False)
    expr_analyzer = ExpressionAnalyzer(debug=False)

    db = load_db_npz(DB_PATH)
    if not db:
        raise RuntimeError(f"No enrolled identities found at {DB_PATH}. Run enroll.py first.")
    matcher = FaceDBMatcher(db=db, dist_thresh=DIST_THRESH)
    print(f"[identity_lock] loaded {len(matcher._names)} identities: {matcher._names}")

    if ENABLE_SERVO:
        print("[identity_lock] ENABLE_SERVO is True but servo wiring isn't implemented in this "
              "version -- see track_and_follow.py for the MQTT/servo control loop.")

    print("Identity Lock running (index 0). Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            H, W = frame.shape[:2]
            vis = frame.copy()

            faces = det.detect(frame, max_faces=1)

            if faces:
                f = faces[0]
                aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                emb = embedder.embed(aligned)
                mr = matcher.match(emb)
                print(f"[identity_lock] name={mr.name} dist={mr.distance:.3f} accepted={mr.accepted}")

                if mr.accepted:
                    # KNOWN person -- locked, show name + expression + blinks
                    color = (0, 255, 0)

                    face_roi = frame[f.y1:f.y2, f.x1:f.x2]
                    expr = expr_analyzer.analyze(face_roi)

                    cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
                    cv2.putText(vis, mr.name, (f.x1, max(0, f.y1 - 40)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    if expr:
                        cv2.putText(vis, expr.expression, (f.x1, max(0, f.y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(vis, f"Blinks: {expr.blink_count}",
                                    (f.x1, min(H - 10, f.y2 + 25)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    # STRANGER -- detected but not locked, no expression analysis
                    color = (0, 0, 255)
                    expr_analyzer.reset_blink_count()  # don't carry a stranger's blink count into the next lock
                    cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
                    cv2.putText(vis, "Stranger", (f.x1, max(0, f.y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Identity Lock (index 0)", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()