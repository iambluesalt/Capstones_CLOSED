"""Evaluate the ASL model on images it was NOT trained on.

Data layouts (auto-detected):
  folder mode   DIR/A/*.jpg, DIR/B/*.jpg, ... DIR/del, DIR/nothing, DIR/space
                (Kaggle train layout, or your own captures from webcam_demo.py --collect)
  flat mode     DIR/A_test.jpg, DIR/nothing_test.jpg, ...  (Kaggle's 28-image test set)

  python evaluate.py --data my_test
  python evaluate.py --data my_test --stress        # also measure robustness
"""
import argparse
import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from asl_common import DEFAULT_MODEL, HERE, LABEL_TO_ID, LABELS, load_image, load_model, predict_probs

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_samples(root):
    root = Path(root)
    samples = []
    folders = [d for d in root.iterdir() if d.is_dir() and d.name in LABEL_TO_ID]
    if folders:
        for d in folders:
            samples += [(p, LABEL_TO_ID[d.name]) for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXT]
    else:
        for p in sorted(root.iterdir()):
            name = p.stem.split("_")[0]
            if p.suffix.lower() in IMG_EXT and name in LABEL_TO_ID:
                samples.append((p, LABEL_TO_ID[name]))
    if not samples:
        raise SystemExit(f"No labelled images found in {root} (see the layouts in --help).")
    return samples


def read_all(samples):
    paths, imgs, y = [], [], []
    for p, label in samples:
        im = load_image(p)
        if im is not None:
            paths.append(p)
            imgs.append(im)
            y.append(label)
    return paths, imgs, np.array(y)


# --- perturbations for --stress: each takes/returns a uint8 BGR image --------------------------
def _affine(im, angle=0.0, shift=0.0, scale=1.0):
    h, w = im.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[:, 2] += (shift * w, shift * h)
    return cv2.warpAffine(im, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def _noise(im, sigma=20, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(im + rng.normal(0, sigma, im.shape), 0, 255).astype(np.uint8)


STRESS = {
    "brighter (+60)": lambda im: cv2.convertScaleAbs(im, alpha=1.0, beta=60),
    "darker (-60)": lambda im: cv2.convertScaleAbs(im, alpha=1.0, beta=-60),
    "low contrast (x0.6)": lambda im: cv2.convertScaleAbs(im, alpha=0.6, beta=20),
    "blur (k=7)": lambda im: cv2.GaussianBlur(im, (7, 7), 0),
    "noise (sigma=20)": _noise,
    "rotate 15 deg": lambda im: _affine(im, angle=15),
    "shift 15%": lambda im: _affine(im, shift=0.15),
    "hand smaller (x0.7)": lambda im: _affine(im, scale=0.7),
    "hand larger (x1.3)": lambda im: _affine(im, scale=1.3),
    "R/B channels swapped": lambda im: im[..., ::-1].copy(),
}


def evaluate(model, imgs, y):
    probs = predict_probs(model, imgs)
    return probs, probs.argmax(1)


def write_report(name, out_dir, paths, y, probs, pred):
    present = sorted(set(y) | set(pred))
    names = [LABELS[i] for i in present]
    top3 = np.mean([t in np.argsort(p)[-3:] for t, p in zip(y, probs)])
    conf = probs.max(1)
    wrong = pred != y
    lines = [
        f"images: {len(y)}   classes present: {len(set(y))}",
        f"accuracy: {np.mean(~wrong):.4f}   top-3: {top3:.4f}   macro-F1: {f1_score(y, pred, average='macro'):.4f}",
        f"mean confidence: right {conf[~wrong].mean():.3f}  wrong {conf[wrong].mean() if wrong.any() else float('nan'):.3f}",
        f"confident errors (wrong and conf >= 0.9): {int(np.sum(wrong & (conf >= 0.9)))}",
        "",
        classification_report(y, pred, labels=present, target_names=names, zero_division=0),
    ]
    cm = confusion_matrix(y, pred, labels=present)
    off = [(cm[i, j], names[i], names[j]) for i in range(len(present)) for j in range(len(present)) if i != j and cm[i, j]]
    lines.append("most confused (count, true -> predicted):")
    lines += [f"  {n:4d}  {t} -> {p}" for n, t, p in sorted(off, reverse=True)[:10]] or ["  none"]
    text = "\n".join(lines)
    print(text)

    (out_dir / f"{name}_report.txt").write_text(text, encoding="utf-8")
    with open(out_dir / f"{name}_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "confidence", "correct"])
        for p, t, q, c in zip(paths, y, pred, conf):
            w.writerow([p, LABELS[t], LABELS[q], f"{c:.4f}", int(t == q)])
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=90)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_confusion.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder-per-class dir or flat dir of <LABEL>_*.jpg")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--out", default=str(HERE / "reports"))
    ap.add_argument("--name", default=None, help="prefix for report files (default: data dir name)")
    ap.add_argument("--stress", action="store_true", help="also re-score under brightness/blur/rotation/... changes")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or Path(args.data).resolve().name
    model = load_model(args.model)
    paths, imgs, y = read_all(collect_samples(args.data))
    probs, pred = evaluate(model, imgs, y)
    write_report(name, out_dir, paths, y, probs, pred)

    if args.stress:
        base = np.mean(pred == y)
        rows = [("clean", base, 0.0)]
        for label, fn in STRESS.items():
            _, p = evaluate(model, [fn(im) for im in imgs], y)
            rows.append((label, np.mean(p == y), np.mean(p == y) - base))
        print("\nrobustness (accuracy under perturbation):")
        with open(out_dir / f"{name}_stress.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["condition", "accuracy", "delta_vs_clean"])
            for label, acc, d in rows:
                print(f"  {label:24s} {acc:6.1%}  ({d:+.1%})")
                w.writerow([label, f"{acc:.4f}", f"{d:+.4f}"])
    print(f"\nreports written to {out_dir}")


if __name__ == "__main__":
    main()
