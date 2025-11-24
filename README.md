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
| `-T`, `--threshold`| `FLOAT` | Manual confidence threshold (0.0 - 1.0). Overrides the model's optimal default. |
| `-k`, `--topk` | `INT` | Output only the top `K` drops per track. If set without a threshold, ignores confidence scores. |
| `-c`, `--csv` | `PATH` | Save predictions to a CSV file. Default: `./model_predictions.csv`. |
| `-t`, `--tag` | N/A | Write drop times (e.g., `DROP_TIME=60.5,124.2`) into the audio file metadata. |

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

## 🧠 How It Works

The system utilizes a three-stage pipeline:

### 1. 🔍 Candidate Generation
Instead of scanning every millisecond, the system uses fast signal processing heuristics to identify potential points of interest:
*   **Bass Boost & Envelope:** Applies a low-shelf filter and scans the volume envelope for sharp energy rises.
*   **Transient Snapping:** Mathematically aligns timestamps to the exact "kick" or transient to ensure rhythmic accuracy.

### 2. 🎛️ Feature Extraction
For each candidate, the model extracts **29 context-aware features**, comparing the audio *after* the impact against the build-up *before* it:

*   **RMS Energy:** Does the volume increase?
*   **Future Energy:** Is there a louder section coming up later?
*   **Grid Alignment:** Does the candidate land on a significant 4, 8, 16, or 32-bar boundary?
*   **Pulse Clarity:** Is there a strong, defined rhythmic pulse, or is the texture messy?
*   **Transient Dominance:** Does the initial "kick" stand out compared to its surroundings?
*   **Bass Ratio:** Is the sound spectrum suddenly dominated by low-frequency energy?
*   **Bass Continuity:** Does the bassline sustain, or does it fade?

### 3. 🤖 Classification (XGBoost)
The core logic is handled by a **Gradient Boosted Decision Tree**.
*   **Optuna Tuning:** Uses Bayesian optimization to find the perfect hyperparameter combination (depth, learning rate, etc.) for maximizing precision.
*   **Dynamic Thresholding:** Automatically calibrates the probability cutoff to maximize the **F1-Score**, rather than using a static 50% default.

## ⚠️ Limitations

*   **Unconventional Structure:** Tracks with irregular time signatures or lack of percussion may yield fewer candidates.
*   **Mastering:** Extreme compression or sparse bass mixing can mask the specific "energy contrast" features the model looks for.
*   **Fake-outs:** "Fake drops" can sometimes trick the model if they are rhythmically similar to a real drop.