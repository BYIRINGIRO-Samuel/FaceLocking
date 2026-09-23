# src/calibrate_expression.py
"""
Live Expression & Eye State Calibration Tool.

Shows raw EAR (Eye Aspect Ratio), mouth corner lift, and mouth width live,
helping you observe how your face maps to:
- Eyes: OPEN vs CLOSED
- Expression: Smiling vs Neutral vs Sad

Run:
    python -m src.calibrate_expression

Keys:
    q : quit
"""
import cv2

from .expression import ExpressionAnalyzer


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[calibrate] Camera 0 failed, trying camera 1...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError("Camera not opened.")

    analyzer = ExpressionAnalyzer(debug=False)

    print("Expression & Eye State calibration tool.")
    print("Try: smiling, looking neutral, drooping corners (sad), closing/opening eyes.")
    print("Press 'q' to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = analyzer.analyze(frame)
        vis = frame.copy()

        if result:
            eye_state = "CLOSED" if result.eyes_closed else "OPEN"
            lines = [
                f"Expression: {result.expression}",
                f"Eyes: {eye_state} (EAR: {result.ear_avg:.3f}, thresh: 0.20)",
                f"Mouth corner lift: {result.mouth_norm_diff:+.3f} (smile >= +0.018, sad < -0.065)",
                f"Mouth width ratio: {result.mouth_width_norm:.3f} (smile >= 0.77)",
                f"Blinks: {result.blink_count}",
            ]
            for i, line in enumerate(lines):
                # Outline
                cv2.putText(vis, line, (10, 32 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis, line, (10, 32 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(vis, "No face detected in crop", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Expression & Eye State Calibration", vis)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()