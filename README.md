# Drop Detector

A machine-learning model designed to detect "drops" (high-energy payoffs) in musical tracks, with a specific optimization for Electronic Dance Music (EDM).

This project iterates upon the work of [Yadati et al. (ISMIR 2014)](https://archives.ismir.net/ismir2014/paper/000297.pdf) regarding content-based drop detection. By utilizing modern signal processing features and XGBoost, this implementation significantly improves detection accuracy:

| Model | F1 Score |
| :--- | :--- |
| Yadati et al. (2014) | 0.71 |
| **Drop Detector (This Repo)** | **0.95** |

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/edm-drop-detector.git
   cd edm-drop-detector
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg:**
   This tool relies on `ffmpeg` for audio processing.
   * **Mac:** `brew install ffmpeg`
   * **Linux:** `sudo apt install ffmpeg`
   * **Windows:** Download binary and add to PATH.

## 🚀 Quick Start: Using the Pre-trained Model

You can use the CLI tool `model_predict.py` to scan folders of audio and detect drops.

### Usage
```bash
python model_predict.py [OPTIONS]
```

### Arguments

| Flag | Argument | Description |
| :--- | :--- | :--- |
| `-f`, `--folder` | `PATH` | **Required.** Path to the root directory containing audio files to scan. |
| `-m`, `--model` | `PATH` | Path to the `.joblib` model file. Defaults to `./model.joblib`. |
| `-t`, `--threshold`| `FLOAT` | Manual confidence threshold (0.0 - 1.0). Overrides the model's optimal default. |
| `-k`, `--topk` | `INT` | Output only the top `K` drops per track. If set without a threshold, ignores confidence scores. |
| `-c`, `--csv` | `[PATH]` | Save predictions to a CSV file. Default: `./model_predictions.csv`. |
| `-T`, `--tag` | N/A | Write drop times (e.g., `DROP_TIME=60.5,124.2`) into the audio file metadata. |

### Example
Scan the `test` folder, save the **top 3** drops that have a confidence **above 90%**, save the results to CSV, and tag the actual audio files:

```bash
python model_predict.py \
  -f "/Users/Admin/Music/Download/test" \
  --csv \
  --topk 3 \
  --threshold 0.90 \
  --tag
```

---

## 🛠️ Advanced: Training Your Own Model

If you wish to retrain the model on your own dataset, follow these steps:

### 1. Data Preparation
Place your raw audio files into the `dataset_train/` folder.

### 2. Labeling
Run the labeling assistant:
```bash
python dataset_build.py
```
This script will iterate through your tracks, propose candidates, and ask you to verify if they are true drops.

### 3. Cleaning
Once labeled, process the data into a clean CSV format for the model:
```bash
python dataset_clean.py
```

### 4. Training
Run the training pipeline:
```bash
python model_build.py
```
This will:
1. Extract features for all labeled candidates.
2. Run Bayesian Hyperparameter Optimization (Optuna).
3. Train an XGBoost classifier.
4. Output the final `model.joblib` file.

---

## 🧠 How It Works

The system operates in three stages: **Candidate Generation**, **Feature Extraction** and **Classification**.

### 1. Candidate Generation
To avoid processing every millisecond of audio, the system first finds "potential" drops based on heuristics:
*   **Bass Boost:** A low-shelf filter is applied to emphasize the "kick."
*   **Envelope Threshold:** It looks for sharp rises in the volume envelope.
*   **Transient Snapping:** Timestamps are mathematically "snapped" to the exact moment of the nearest transient (beat).

### 2. Feature Extraction
For every candidate, the model analyzes the audio context (comparing the window *after* the drop to the windows *before* it). The model uses **29 specific features**, grouped as follows:

*   **RMS Energy Differences:** Compares volume intensity of the impact vs. the build-up (Short, Medium, and Long lookback windows).
*   **Future Energy Dominance:** Is there a louder section coming up later?
*   **Grid Alignment:** Uses beat tracking to determine if the drop lands on a significant musical phrase (4, 8, 16, or 32-bar chunks).
*   **Pulse Clarity:** Analysis of rhythmic strength, ie. Is there a clear beat pattern?.
*   **Transient Dominance:** How significant is the impact transient compared to its surroundings?
*   **Bass Ratio:** The proportion of low-frequency energy compared to the total spectrum.
*   **Bass Continuity:** Does the bass sustain after the impact, or does it fade?

### 3. Classification
The heart of the detector is a **Gradient Boosted Decision Tree (XGBoost)** classifier. Unlike simple volume-based detection, this model learns complex, non-linear relationships between the extracted features.

*   **Bayesian Optimization:** The training pipeline uses **Optuna** to perform automated hyperparameter tuning. It iteratively tests combinations of tree depth, learning rates, and regularization to maximize the **Precision-Recall AUC**.
*   **Dynamic Thresholding:** Instead of a fixed 50% probability cutoff, the training script calculates the specific probability threshold that maximizes the **F1-Score** on the test set. This metadata is saved with the model to ensure the CLI tool uses the exact same sensitivity standards as the training environment.

## ⚠️ Limitations

*   **Unconventional Structures:** Tracks with non-standard time signatures, or lack of percussion may not yield good results.
*   **Mastering Issues:** Tracks with very sparse bass or extreme dynamic range compression might reduce the specific "energy contrast" features the model relies on.
*   **False Positives:** Heavy breakdowns or "fake drops" can sometimes trick the model if they are rhythmically similar to a real drop.