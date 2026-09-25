"""Live ASL prediction from a webcam (or video file), and a tool to build your own test set.

  python webcam_demo.py                       # live prediction, press q or Esc to quit
  python webcam_demo.py --collect             # press a letter key to save the frame as my_test/<LETTER>/
                                              #   1 = del, 2 = nothing, 3 = space, Esc = quit
  python webcam_demo.py --source clip.mp4 --no-display     # run on a video file, print predictions

The model saw whole webcam frames with the hand filling most of the picture, so the default
region of interest is the full centre square of the frame. Then score what you collected with:
  python evaluate.py --data my_test --stress
"""
import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from asl_common import DEFAULT_MODEL, HERE, LABELS, load_model, predict_probs

COLLECT_KEYS = {ord("1"): "del", ord("2"): "nothing", ord("3"): "space"}


def center_square(frame, frac):
    h, w = frame.shape[:2]
    side = int(min(h, w) * frac)
    x0, y0 = (w - side) // 2, (h - side) // 2
    return frame[y0 : y0 + side, x0 : x0 + side], (x0, y0, x0 + side, y0 + side)


def key_to_label(key):
    if key in COLLECT_KEYS:
        return COLLECT_KEYS[key]
    if ord("a") <= key <= ord("z"):
        return chr(key).upper()
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--source", default="0", help="camera index or video file path (default 0)")
    ap.add_argument("--roi", type=float, default=1.0, help="fraction of the centre square to use (default 1.0)")
    ap.add_argument("--smooth", type=int, default=5, help="average probabilities over the last N frames")
    ap.add_argument("--min-conf", type=float, default=0.6, help="show '?' below this confidence")
    ap.add_argument("--mirror", action="store_true", help="flip the frame horizontally (selfie view)")
    ap.add_argument("--collect", action="store_true", help="save labelled frames instead of predicting")
    ap.add_argument("--save-dir", default=str(HERE / "my_test"))
    ap.add_argument("--no-display", action="store_true", help="no window; print predictions (for files/CI)")
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source {args.source!r}")
    model = None if args.collect else load_model(args.model)
    history = deque(maxlen=args.smooth)
    save_dir = Path(args.save_dir)
    counts, last_saved, frame_no = {}, "", 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if args.mirror:
            frame = cv2.flip(frame, 1)  # off by default: handedness may matter to the model
        roi, (x0, y0, x1, y1) = center_square(frame, args.roi)

        text = ""
        if model is not None:
            history.append(predict_probs(model, [roi])[0])
            probs = np.mean(history, axis=0)
            top = np.argsort(probs)[::-1][:3]
            label = LABELS[top[0]] if probs[top[0]] >= args.min_conf else "?"
            text = f"{label}  ({probs[top[0]]:.0%})   " + "  ".join(f"{LABELS[i]} {probs[i]:.0%}" for i in top[1:])
            if args.no_display:
                print(f"frame {frame_no}: {text}")

        if args.no_display:
            continue
        view = frame.copy()
        cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 0), 2)
        if args.collect:
            text = f"collect: press a letter (1=del 2=nothing 3=space). saved: {sum(counts.values())} last: {last_saved}"
        cv2.putText(view, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("ASL", view)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or (key == ord("q") and not args.collect):
            break
        if args.collect and key != 255:
            label = key_to_label(key)
            if label:
                d = save_dir / label
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{label}_{int(time.time() * 1000)}.jpg"), roi)
                counts[label] = counts.get(label, 0) + 1
                last_saved = f"{label} ({counts[label]})"

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    if counts:
        print("saved:", dict(sorted(counts.items())), "->", save_dir)


if __name__ == "__main__":
    main()
