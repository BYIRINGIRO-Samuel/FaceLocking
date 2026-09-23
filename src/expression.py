# src/expression.py
"""
Lightweight expression + blink/eye state estimation using MediaPipe FaceMesh
landmark geometry.

Classifies: "Smiling" / "Neutral" / "Sad"
Tracks:
  - Eyes state: Open vs Closed (via Eye Aspect Ratio - EAR)
  - Running blink count
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

# Mouth landmarks
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
MOUTH_TOP, MOUTH_BOTTOM = 13, 14

# --- Tuning Thresholds ---
BLINK_EAR_THRESH = 0.20       # avg EAR below this = eyes considered closed
BLINK_MIN_CONSEC_FRAMES = 2   # frames eyes must stay closed to count as a full blink

# Smile & Sad heuristics (normalized by interocular outer-eye distance)
SMILE_LIFT_THRESH = 0.018     # mouth corner lift above mid-lip
SMILE_WIDTH_THRESH = 0.77     # mouth width ratio (neutral is ~0.65-0.72, smiling is >0.76)
SAD_LIFT_THRESH = -0.065      # corner droop threshold for sad (below -0.065)


@dataclass
class ExpressionResult:
    expression: str          # "Smiling" / "Neutral" / "Sad"
    eyes_closed: bool        # True if eyes are closed, False if open
    ear_avg: float           # raw Eye Aspect Ratio
    mouth_norm_diff: float   # corner lift ratio
    mouth_width_norm: float  # mouth width ratio
    blink_count: int
    is_blinking: bool


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))


class ExpressionAnalyzer:
    """
    Runs MediaPipe FaceMesh on a face ROI (crop) and estimates
    expression + eye state (open/closed) + blink count.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        if mp is None:
            raise RuntimeError(
                f"mediapipe import failed: {_MP_IMPORT_ERROR}\n"
                f"Install: pip install mediapipe==0.10.21"
            )
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
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

        # --- Eye Aspect Ratio (Open / Closed Eyes & Blink Detection) ---
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

        # --- Mouth geometry (Smile / Neutral / Sad) ---
        eye_dist = _dist(pt(L_EYE_OUTER), pt(R_EYE_OUTER))  # interocular distance for normalization
        mid_lip_y = (pt(MOUTH_TOP)[1] + pt(MOUTH_BOTTOM)[1]) / 2.0
        corner_avg_y = (pt(MOUTH_LEFT)[1] + pt(MOUTH_RIGHT)[1]) / 2.0

        # Positive = corners raised above mid-lip (smile), Negative = corners drooped (sad)
        mouth_norm_diff = (mid_lip_y - corner_avg_y) / max(1e-6, eye_dist)

        mouth_width = _dist(pt(MOUTH_LEFT), pt(MOUTH_RIGHT))
        mouth_width_norm = mouth_width / max(1e-6, eye_dist)

        # Multi-signal smile detection:
        # 1. Clear corner lift
        # 2. Significant mouth widening
        # 3. Combined subtle lift + moderate widening
        is_smiling = (
            (mouth_norm_diff >= SMILE_LIFT_THRESH) or
            (mouth_width_norm >= SMILE_WIDTH_THRESH) or
            (mouth_norm_diff >= 0.008 and mouth_width_norm >= 0.73)
        )

        is_sad = (mouth_norm_diff < SAD_LIFT_THRESH)

        if is_smiling:
            expression = "Smiling"
        elif is_sad:
            expression = "Sad"
        else:
            expression = "Neutral"

        return ExpressionResult(
            expression=expression,
            eyes_closed=is_closed_now,
            ear_avg=ear_avg,
            mouth_norm_diff=mouth_norm_diff,
            mouth_width_norm=mouth_width_norm,
            blink_count=self._blink_count,
            is_blinking=is_closed_now,
        )