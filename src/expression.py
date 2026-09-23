# src/expression.py
"""
Lightweight expression + blink estimation using MediaPipe FaceMesh landmark
geometry. This is a HEURISTIC, not a trained classifier -- it compares
landmark distances against tunable thresholds. Good enough for a live demo,
but the thresholds below are starting points and will likely need tuning
for your specific face/camera/lighting (see calibrate_expression.py).

Classifies: "Smiling" / "Neutral" / "Sad"
Tracks: running blink count (via eye-aspect-ratio dips)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as e:
    mp = None
    _MP_IMPORT_ERROR = e


# Landmark indices (MediaPipe FaceMesh, refine_landmarks=True)
# Eyes (simplified 4-point EAR -- outer corner, inner corner, top lid, bottom lid)
L_EYE_OUTER, L_EYE_INNER, L_EYE_TOP, L_EYE_BOTTOM = 33, 133, 159, 145
R_EYE_OUTER, R_EYE_INNER, R_EYE_TOP, R_EYE_BOTTOM = 263, 362, 386, 374

# Mouth
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
MOUTH_TOP, MOUTH_BOTTOM = 13, 14

# Tuning -- ADJUST THESE after watching your own printed values (see calibrate_expression.py)
BLINK_EAR_THRESH = 0.21       # avg EAR below this = eyes considered closed
BLINK_MIN_CONSEC_FRAMES = 2   # must stay closed this many frames to count (avoids false triggers)
SMILE_THRESH = 0.06           # normalized corner-lift above this = smiling (signal 1)
SAD_THRESH = 0.04             # normalized corner-drop below this = sad
SMILE_WIDTH_THRESH = 1.05     # mouth width / interocular distance above this = smiling (signal 2)


@dataclass
class ExpressionResult:
    expression: str          # "Smiling" / "Neutral" / "Sad"
    ear_avg: float           # raw value, useful for calibrate_expression.py tuning
    mouth_norm_diff: float   # raw value, useful for calibrate_expression.py tuning
    mouth_width_norm: float  # raw value, useful for calibrate_expression.py tuning
    blink_count: int
    is_blinking: bool


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))


class ExpressionAnalyzer:
    """
    Runs its own MediaPipe FaceMesh pass on a face ROI (crop) and estimates
    expression + blink state. Keep one instance alive across frames -- it
    tracks blink count internally.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        if mp is None:
            raise RuntimeError(
                f"mediapipe import failed: {_MP_IMPORT_ERROR}\n"
                f"Install: pip install mediapipe==0.10.21"
            )
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._blink_count = 0
        self._consec_closed = 0
        self._was_closed = False

    def reset_blink_count(self):
        self._blink_count = 0
        self._consec_closed = 0
        self._was_closed = False

    def analyze(self, roi_bgr: np.ndarray) -> Optional[ExpressionResult]:
        H, W = roi_bgr.shape[:2]
        if H < 20 or W < 20:
            return None

        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None

        lm = res.multi_face_landmarks[0].landmark

        def pt(i):
            p = lm[i]
            return (p.x * W, p.y * H)

        # --- Eye Aspect Ratio (blink detection) ---
        l_ear = _dist(pt(L_EYE_TOP), pt(L_EYE_BOTTOM)) / max(1e-6, _dist(pt(L_EYE_OUTER), pt(L_EYE_INNER)))
        r_ear = _dist(pt(R_EYE_TOP), pt(R_EYE_BOTTOM)) / max(1e-6, _dist(pt(R_EYE_OUTER), pt(R_EYE_INNER)))
        ear_avg = (l_ear + r_ear) / 2.0

        is_closed_now = ear_avg < BLINK_EAR_THRESH
        if is_closed_now:
            self._consec_closed += 1
        else:
            if self._was_closed and self._consec_closed >= BLINK_MIN_CONSEC_FRAMES:
                self._blink_count += 1
            self._consec_closed = 0
        self._was_closed = is_closed_now

        # --- Mouth shape (smile/neutral/sad heuristic) ---
        eye_dist = _dist(pt(L_EYE_OUTER), pt(R_EYE_OUTER))  # interocular distance, for scale normalization
        mid_lip_y = (pt(MOUTH_TOP)[1] + pt(MOUTH_BOTTOM)[1]) / 2.0
        corner_avg_y = (pt(MOUTH_LEFT)[1] + pt(MOUTH_RIGHT)[1]) / 2.0
        # positive = corners raised above mid-lip (smile signal), negative = corners drooped (sad)
        mouth_norm_diff = (mid_lip_y - corner_avg_y) / max(1e-6, eye_dist)

        mouth_width = _dist(pt(MOUTH_LEFT), pt(MOUTH_RIGHT))
        mouth_width_norm = mouth_width / max(1e-6, eye_dist)  # smile signal: mouth widens when smiling

        if mouth_norm_diff < -SAD_THRESH:
            expression = "Sad"
        elif mouth_norm_diff > SMILE_THRESH or mouth_width_norm > SMILE_WIDTH_THRESH:
            expression = "Smiling"
        else:
            expression = "Neutral"

        return ExpressionResult(
            expression=expression,
            ear_avg=ear_avg,
            mouth_norm_diff=mouth_norm_diff,
            mouth_width_norm=mouth_width_norm,
            blink_count=self._blink_count,
            is_blinking=is_closed_now,
        )