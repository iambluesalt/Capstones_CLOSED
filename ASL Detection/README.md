# ASL Alphabet Detection

A small CNN (3 conv blocks, 1.28M parameters, 64×64 RGB input) that classifies a photo of a hand into one of 29 classes: A–Z, `del`, `nothing`, `space`. Trained in `asl-test.ipynb` on Kaggle's ASL Alphabet dataset (87,000 images).

## What the numbers mean

| Number | Where | What it is |
|---|---|---|
| 91% | old README | matches the last-epoch *training* accuracy (91.4%, dropout on) |
| 98.76% | notebook | accuracy on a random 30% split. The **same** split was used for early stopping, so it is not an untouched test set |
| 26/29 | `reports/smoke_29_tiles_*` | smoke test, see below. Not an accuracy estimate |

**98.76% is an upper bound, not the expected real-world accuracy.**

* The Kaggle images look like consecutive frames from short clips, one room, one backdrop, a few hands (visible in the notebook's own image grids). A random split puts near-identical neighbours on both sides, so the score partly measures memorisation. (Not checked against the raw dataset, which is not in this repo.)
* The 28 official test images that ship with the dataset were never used.
* Nothing was tried on a different camera, background, lighting, or person, which is what "deployment" means.

## How to actually test the model

Run these from easiest to most honest. Numbers from step 1 say the code works; only steps 2–3 say the model works.

**1. Sanity check on the dataset's own test images** (28 files like `A_test.jpg`; needs the Kaggle download):

```bash
python evaluate.py --data path/to/asl_alphabet_test/asl_alphabet_test
```

Too small (1 image per class) for accuracy claims. It only proves that loading and preprocessing are wired correctly.

**2. Build your own test set, then score it.** This is the real test. Use a different room, lighting and, ideally, a different person than the training data:

```bash
python webcam_demo.py --collect            # press a letter to save the frame; 1=del 2=nothing 3=space; Esc quits
python evaluate.py --data my_test --stress
```

Aim for at least 20 images per letter, in varied positions. With that many, a per-class result like 18/20 has a wide error bar, so read the overall accuracy and the per-class recall table together. `evaluate.py` writes to `reports/`: accuracy, top-3, macro-F1, per-class table, most-confused pairs, confident errors, a confusion matrix, and a CSV of every prediction. `--stress` re-scores the same images under brightness, contrast, blur, noise, rotation, shift, zoom changes, and swapped colour channels.

**3. Look at it live:**

```bash
python webcam_demo.py                      # q or Esc quits; --mirror for selfie view; --roi 0.8 to crop tighter
```

Predictions are averaged over the last 5 frames and shown as `?` under 60% confidence.

**4. Best of all:** test on a public ASL-alphabet dataset the model has never seen (different signers/backgrounds), laid out one folder per class.

`J` and `Z` are motion signs in real ASL. A single still image can only capture a snapshot of them, so expect those two to stay weak.

## What I verified (smoke test only)

The notebook embeds a 29-image grid, one training-set image per class. I cropped it into `A/…/space` folders and ran `evaluate.py --stress` (`reports/smoke_29_tiles_*`). These are re-rendered training-distribution images and n = 29, so treat the sizes of the drops as rough:

| Condition | Accuracy |
|---|---|
| clean | 89.7% (26/29; errors: del→P, X→S, U→R) |
| brighter / darker | 51.7% / 65.5% |
| low contrast | 58.6% |
| rotate 15° | 48.3% |
| shift 15% | 24.1% |
| hand smaller (×0.7) | 31.0% |
| R/B channels swapped | 27.6% |

Takeaways: the pipeline reproduces the notebook's behaviour, the model is **fragile to framing, lighting and hand position**, and the notebook's preprocessing quirk matters: it feeds **BGR** images (OpenCV order, never converted to RGB). Every script here goes through `asl_common.preprocess_bgr` so the model sees what it was trained on. Using RGB inputs, e.g. in a web/mobile app, silently costs ~60 points on this test.

## Retraining (not yet run on the real data)

```bash
python train.py --data "<...>/asl_alphabet_train/asl_alphabet_train"
```

Fixes over the notebook: contiguous train | gap | val | gap | test blocks per class (neighbouring frames never straddle a split), a test block scored once at the end, augmentation (rotation, shift, zoom, brightness, contrast) to address the fragility above, `restore_best_weights`, and an LR schedule that reacts to validation loss. Output: `ASL_v2.keras` and `reports/train_test_block_report.txt`.

`train.py` was smoke-tested end to end on a synthetic dataset (split sizes, augmentation, training, saving), not on the Kaggle data, so no accuracy is claimed for it. Score the retrained model with step 2 above, using the same `evaluate.py`: `python evaluate.py --data my_test --model ASL_v2.keras`.

## Files

```
asl-test.ipynb            original training notebook (unchanged)
ASL_fudge_fantastic.h5    trained model (notebook saved it as ASL_fudge_fantastic_1.h5)
asl_common.py             labels + the exact training preprocessing + model loading
evaluate.py               score a labelled image folder, optional --stress robustness test
webcam_demo.py            live prediction, or --collect to build your own test set
train.py                  leak-resistant, augmented retraining script
reports/                  smoke-test results
```

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe evaluate.py --data my_test --stress
```

## Next steps

* Run steps 2–3 above and replace the 98.76% headline with the number from your own captures.
* Retrain with `train.py` on Kaggle (GPU) and compare on the same captures.
* For a real product: crop to the hand first (e.g. MediaPipe Hands) so framing and background stop mattering, then classify the crop.
