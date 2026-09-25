"""Retrain the ASL CNN with a leak-resistant split and data augmentation.

  python train.py --data "<...>/asl_alphabet_train/asl_alphabet_train"

Why not the notebook's random 70/30 split: the Kaggle images are consecutive frames from short
clips (A1.jpg ... A3000.jpg), so a random split puts near-identical neighbours on both sides and
the score mostly measures memorisation. Here every class is cut into contiguous blocks
(train | gap | val | gap | test), so neighbouring frames never straddle a split. The test block
is scored once, at the end. Same-signer / same-room bias remains: score on your own captures too
(webcam_demo.py --collect, then evaluate.py).
"""
import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from asl_common import HERE, LABEL_TO_ID, LABELS, load_image, preprocess_bgr

IMG_EXT = {".jpg", ".jpeg", ".png"}


def natural_key(p):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def block_split(files, gap, frac_val=0.15, frac_test=0.15):
    n = len(files)
    n_val, n_test = int(n * frac_val), int(n * frac_test)
    n_train = n - n_val - n_test - 2 * gap
    if n_train < 1:
        raise SystemExit(f"Not enough images per class ({n}) for gap={gap}; lower --gap or raise --limit.")
    val_start = n_train + gap
    test_start = val_start + n_val + gap
    return files[:n_train], files[val_start : val_start + n_val], files[test_start:]


def load_files(files, label, workers=8):
    def one(p):
        im = load_image(p)
        return None if im is None else preprocess_bgr(im)

    with ThreadPoolExecutor(workers) as ex:
        imgs = [im for im in ex.map(one, files) if im is not None]
    return imgs, [label] * len(imgs)


def build_model(num_classes=len(LABELS)):
    from tensorflow import keras
    from tensorflow.keras import layers as L

    def block(f):
        return [L.Conv2D(f, 3), L.BatchNormalization(), L.Activation("relu"), L.MaxPooling2D(2)]

    return keras.Sequential(
        [
            L.Input((64, 64, 3)),
            # augmentation is active only during fit(); disabled at inference
            L.RandomRotation(0.04),
            L.RandomTranslation(0.1, 0.1),
            L.RandomZoom(0.2),
            L.RandomBrightness(0.2, value_range=(0, 1)),
            L.RandomContrast(0.2, value_range=(0, 1)),
            *block(32),
            *block(64),
            *block(128),
            L.Flatten(),
            L.Dense(256, activation="relu"),
            L.Dropout(0.5),
            L.Dense(num_classes, activation="softmax"),
        ]
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Kaggle asl_alphabet_train folder (one sub-folder per class)")
    ap.add_argument("--out", default=str(HERE / "ASL_v2.keras"))
    ap.add_argument("--report-dir", default=str(HERE / "reports"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--gap", type=int, default=30, help="frames dropped between train/val/test blocks")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N images per class (smoke tests)")
    args = ap.parse_args()

    from tensorflow import keras

    root = Path(args.data)
    missing = [c for c in LABELS if not (root / c).is_dir()]
    if missing:
        raise SystemExit(f"Missing class folders under {root}: {missing}")

    parts = {"train": ([], []), "val": ([], []), "test": ([], [])}
    for name in LABELS:
        files = sorted((p for p in (root / name).iterdir() if p.suffix.lower() in IMG_EXT), key=natural_key)
        if args.limit:
            files = files[: args.limit]
        for split, chunk in zip(parts, block_split(files, args.gap)):
            x, y = load_files(chunk, LABEL_TO_ID[name])
            parts[split][0].extend(x)
            parts[split][1].extend(y)
        print(f"loaded {name}", end="\r")
    (Xtr, ytr), (Xva, yva), (Xte, yte) = [(np.stack(x), np.array(y)) for x, y in parts.values()]
    print(f"\ntrain {len(ytr)}  val {len(yva)}  test {len(yte)}")

    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        Xtr,
        ytr,
        validation_data=(Xva, yva),
        epochs=args.epochs,
        batch_size=args.batch,
        verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
        ],
    )
    model.save(args.out)

    pred = model.predict(Xte, batch_size=256, verbose=0).argmax(1)
    report = f"held-out test block: n={len(yte)} accuracy={np.mean(pred == yte):.4f}\n\n" + classification_report(
        yte, pred, labels=range(len(LABELS)), target_names=LABELS, zero_division=0
    )
    print(report)
    Path(args.report_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.report_dir) / "train_test_block_report.txt").write_text(report, encoding="utf-8")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
