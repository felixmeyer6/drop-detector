# EDM Drop Detector
Finds the main drop (highest-energy payoff) in an EDM track.

This project builds upon the work of [Yadati et al. (ISMIR 2014)](https://archives.ismir.net/ismir2014/paper/000297.pdf) on content-based drop detection and makes their method into a practical, notebook-driven pipeline.

## How It Works
A lightweight pipeline turns raw audio into 1–2 predicted drop times:

| Stage | What It Does |
|-------|--------------|
| Candidate Proposal | Splits the track into musically plausible boundary points (phrase/structure changes). |
| Context & Features | Extracts a window around each candidate (with low-end emphasis) and builds simple statistics/features capturing energy lift and contrast. |
| Scoring | Assigns a score to each candidate, ranking the most likely drops. |
| Classsifying | Returns the top 1–2 moments. Evaluation uses F1@2: a prediction is correct if either of the 2 returned times hits within the tolerance window around the ground truth. |

## Using a Pretrained Model
* Load Model: `joblib.load("drop_detector_model.joblib")`
* Run Inference: `find_drop_times("path/to/track.wav")`

**Requirements:** Notebook + `.joblib` bundle.

## Training Your Own Model

### 1. Prepare Dataset
| Cell | Details |
|------|---------|
| Configure Paths | Set `BASE_DIR` to your dataset root. |
| Folder Layout | `dataset/track1.wav`, `dataset/track2.mp3`, ... |
| Load CSV Labels | Sidecar CSV: `filename → drop time` (MM:SS or seconds). |
| Read Labels from Metadata | MP3: ID3 TXXX with `desc="DROP_TIME"`; WAV: custom "drop" chunk (float seconds). |
| Build Training Pairs | Produces `pairs_all` (filepaths ↔ drop times). |

### 2. Hyperparameter Tuning
| Cell | Details |
|------|---------|
| Run Hyperparameter Sweep | `sweep_leave_tracks_out(...)` |
| Inspect & Select Best Config | Sort by `F1@2_val` and pick the winner. |

### 3. Train & Export
| Cell | Details |
|------|---------|
| Train on All Labeled Tracks | `train_final_from_pairs(pairs_all, ...)` |
| Export Bundle | Save to `drop_detector_model.joblib`. |

### 4. Predict Drops
| Cell | Details |
|------|---------|
| Load Trained/Pretrained Model Bundle | `joblib.load("drop_detector_model.joblib")` |
| Run Inference | `find_drop_times("my_new_track.wav")` (returns top-2 `{time_s, score}`). |

## Labels & Metrics
- **Ground-truth labels:** One primary “most important” drop per track (CSV or embedded metadata).
- **Metric:** F1@2. A prediction is correct if either of the top-2 times is within the tolerance window around the labeled drop.

## Limitations
- Highly unconventional structures can confuse candidate proposal.
- Very sparse bass or extreme mastering can reduce the “energy lift” signal.

## Safety & Respect
Music is made by people. If this helps you perform, curate, or create, please give credit where it’s due and support the artists you rely on.
