# src/calibrate_expression.py
"""
Run this BEFORE the demo to tune the thresholds in src/expression.py.

Shows the raw ear_avg and mouth_norm_diff values live so you can see what
YOUR neutral/smile/sad/blink actually look like, then adjust
BLINK_EAR_THRESH / SMILE_THRESH / SAD_THRESH in expression.py accordingly.

Run:
    python -m src.calibrate_expression

Keys:
    q : quit
"""
import cv2

from .expression import ExpressionAnalyzer


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # laptop webcam -- change to 0 if calibrating on turret cam
    if not cap.isOpened():
        raise RuntimeError("Camera not opened.")

    analyzer = ExpressionAnalyzer(debug=False)

    print("Expression calibration. Try neutral / smiling / sad / blinking on purpose.")
    print("Watch the printed values -- use them to set thresholds in expression.py.")
    print("Press 'q' to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = analyzer.analyze(frame)
        vis = frame.copy()

        if result:
            lines = [
                f"expression: {result.expression}",
                f"ear_avg: {result.ear_avg:.3f}  (blink thresh: 0.21)",
                f"mouth_norm_diff: {result.mouth_norm_diff:+.3f}  (smile>0.06, sad<-0.04)",
                f"mouth_width_norm: {result.mouth_width_norm:.3f}  (smile>1.05)",
                f"blink_count: {result.blink_count}",
                f"is_blinking: {result.is_blinking}",
            ]
            for i, line in enumerate(lines):
                cv2.putText(vis, line, (10, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "no face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Expression Calibration", vis)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()