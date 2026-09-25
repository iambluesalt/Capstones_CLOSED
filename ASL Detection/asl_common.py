"""Shared preprocessing and model loading for the ASL alphabet CNN.

The notebook trained on `cv2.imread` (BGR, never converted to RGB) followed by
`skimage.transform.resize` (float in [0, 1]). Inference must do exactly the same
or accuracy silently drops, so every script goes through `preprocess_bgr`.
"""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
from skimage.transform import resize

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "ASL_fudge_fantastic.h5"
IMAGE_SIZE = 64
# Index order used by the notebook's label_map (A=0 ... Z=25, del, nothing, space).
LABELS = [*"ABCDEFGHIJKLMNOPQRSTUVWXYZ", "del", "nothing", "space"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}


def preprocess_bgr(img_bgr):
    """uint8 BGR image of any size -> float32 (64, 64, 3) in [0, 1]."""
    out = resize(img_bgr, (IMAGE_SIZE, IMAGE_SIZE, 3))
    return out.astype(np.float32)


def load_image(path):
    """Read an image file as the notebook did (BGR). Returns None if unreadable."""
    return cv2.imread(str(path))


def load_model(path=DEFAULT_MODEL):
    from tensorflow import keras

    return keras.models.load_model(path, compile=False)


def predict_probs(model, images_bgr, batch_size=256):
    """List/array of uint8 BGR images -> (n, 29) softmax probabilities."""
    batch = np.stack([preprocess_bgr(im) for im in images_bgr])
    return model.predict(batch, batch_size=batch_size, verbose=0)
