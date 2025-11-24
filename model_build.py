import sys
import warnings
from pathlib import Path

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedKFold,
    cross_val_score,
)
from threadpoolctl import threadpool_limits
from tqdm import tqdm

# Ensure local modules can be imported
sys.path.append(str(Path(__file__).parent))

from processors import (
    features_batch,
    generate_candidates,
    get_score_at_time,
)
from utilities import (
    compute_envelope,
    low_shelf,
    snap_time,
)

warnings.filterwarnings("ignore")


# Init
CSV_PATH = Path("./dataset_train_clean.csv")
MUSIC_PATH = Path("./dataset_train")

CONF = {
    "sr": 22050,
    "min_distance": 2.0,  # Min seconds between candidates
    "match_tolerance": 0.5,  # Seconds +/- to match label
    "score_lookback": 30,  # Frames for score calculation
    "snap_window": 2.0,  # Window for beat snapping
    "intro_s": 20,  # No drops in the first 20s
    "outro_s": 20,  # No drops in the last 20s
}


# Helpers
def build_track(fname, fpath, positive_times, CONF, intro_s=20, outro_s=20):
    """
    1. Finds candidates.
    2. Labels them.
    3. Injects missing positives.
    4. Filter negatives scored lower than the lowest positive.
    """
    candidates = generate_candidates(fpath, CONF)

    if not candidates and not positive_times.size:
        return []

    # Label Existing Candidates
    labeled_rows = []
    matched_truth_indices = set()

    y_ref = candidates[0]["y_ref"] if candidates else None
    env_ref = candidates[0]["env"] if candidates else None
    fps_ref = candidates[0]["fps"] if candidates else None

    if y_ref is None:
        try:
            y_ref, _ = librosa.load(fpath, sr=CONF["sr"])
            y_boosted = low_shelf(y_ref, CONF["sr"], cutoff=600, gain_db=6.0)
            env_ref, fps_ref = compute_envelope(y_boosted, CONF["sr"])
        except Exception:
            return []

    for cand in candidates:
        c_time = cand["time"]
        c_score = cand["score"]
        label = 0

        if len(positive_times) > 0:
            distances = np.abs(positive_times - c_time)
            min_dist_idx = np.argmin(distances)
            if distances[min_dist_idx] <= CONF["match_tolerance"]:
                label = 1
                matched_truth_indices.add(min_dist_idx)

        labeled_rows.append(
            {
                "filename": fname,
                "fpath": str(fpath),
                "time": c_time,
                "score": c_score,
                "label": label,
            }
        )

    # Inject Missing Positives
    track_duration = len(y_ref) / CONF["sr"]
    for i, p_time in enumerate(positive_times):
        if i not in matched_truth_indices:
            if p_time < intro_s or p_time > (track_duration - outro_s):
                continue

            snapped_p = snap_time(p_time, y_ref, CONF)
            score = get_score_at_time(snapped_p, env_ref, fps_ref, CONF)

            labeled_rows.append(
                {
                    "filename": fname,
                    "fpath": str(fpath),
                    "time": snapped_p,
                    "score": score,
                    "label": 1,
                }
            )

    # Dynamic Threshold Filtering
    df_track = pd.DataFrame(labeled_rows)
    if df_track.empty:
        return []

    positives = df_track[df_track.label == 1]
    negatives = df_track[df_track.label == 0]

    # Remove negatives that are weaker than the weakest drop
    if not positives.empty:
        min_drop_score = positives["score"].min()
        negatives_filtered = negatives[negatives["score"] >= min_drop_score]
        negatives = negatives_filtered

    final_df = pd.concat([positives, negatives])

    return final_df.to_dict("records")


def build_dataset(CONF):
    """
    Loads ground truth from CSV, maps audio files, and executes parallel track processing.
    Aggregates positive (drops) and negative examples into a single DataFrame.
    """
    if not CSV_PATH.exists():
        print(f"Could not find {CSV_PATH}")
        return

    df_labels = pd.read_csv(CSV_PATH)

    if "finished" not in df_labels.columns:
        print("CSV missing 'finished' column.")
        return

    finished_files_df = df_labels[df_labels["finished"]]
    finished_filenames = finished_files_df["filename"].unique()

    print(f"Found {len(finished_filenames)} finished tracks.")

    print("\nIndexing audio files...")
    audio_files_map = {p.name: p for p in MUSIC_PATH.rglob("*") if p.is_file()}

    ground_truth_map = {}
    for fname in finished_filenames:
        matches = finished_files_df[finished_files_df["filename"] == fname]
        ground_truth_map[fname] = matches[matches["label"] == 1]["timestamp"].values

    tasks = []
    for fname in finished_filenames:
        if fname in audio_files_map:
            tasks.append(
                (
                    fname,
                    audio_files_map[fname],
                    ground_truth_map.get(fname, np.array([])),
                    CONF,
                )
            )

    print(f"Computing candidates for {len(tasks)} files...")

    with threadpool_limits(limits=1, user_api="blas"):
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(build_track)(fn, fp, times, CONF)
            for fn, fp, times, conf in tqdm(tasks)
        )

    dataset_rows = [row for batch in results for row in batch]

    if not dataset_rows:
        print("No data generated.")
        return

    df_final = pd.DataFrame(dataset_rows)
    df_final = df_final.sort_values(by=["filename", "time"], ascending=[True, True])

    print("\nDataset Build Complete.")
    print("-" * 30)
    print(f"Total Rows: {len(df_final)}")
    print(f"Positives: {len(df_final[df_final.label == 1])}")
    print(f"Negatives: {len(df_final[df_final.label == 0])}")
    print("-" * 30)

    return df_final


def split(dataset):
    """Splits the dataset into Train and Test sets, grouping by track."""
    groups = dataset["filename"]
    y = dataset["label"]
    X = dataset.drop(columns=["label"])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20)

    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    print("\nSplits (per track)")

    train_files = X_train["filename"].nunique()
    test_files = X_test["filename"].nunique()
    total_files = dataset["filename"].nunique()

    print("-" * 30)
    print(
        f"Train Set: {len(X_train):>5} rows | {train_files:>3} tracks ({train_files / total_files:.1%})"
    )
    print(
        f"Test Set:  {len(X_test):>5} rows | {test_files:>3} tracks ({test_files / total_files:.1%})"
    )
    print("-" * 30)

    # Leakage verification
    train_set = set(X_train["filename"].unique())
    test_set = set(X_test["filename"].unique())

    leakage = train_set.intersection(test_set)

    if leakage:
        print(f"Data leakage detected! Files found in multiple sets: {leakage}")

    X_train = X_train.drop(columns=["filename", "fpath"])
    X_test = X_test.drop(columns=["filename", "fpath"])

    return (
        (X_train, y_train),
        (X_test, y_test),
        (X.drop(columns=["filename", "fpath"]), y),
    )


def xgb_objective(trial, X, y):
    """Optuna objective function using Stratified CV on training data."""
    ratio = float(np.sum(y == 0)) / np.sum(y == 1)

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight", ratio * 0.8, ratio * 1.2
        ),
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "n_jobs": -1,
    }

    model = xgb.XGBClassifier(**params)

    # 5-Fold Stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    scores = cross_val_score(model, X, y, cv=cv, scoring="average_precision")

    return scores.mean()


def run_optimization(X, y, n_trials=30):
    """Runs the Optuna study."""
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: xgb_objective(trial, X, y), n_trials=n_trials)

    print(f"Best CV PR-AUC: {study.best_value:.4f}")
    return study.best_params


def train_and_evaluate(params, X_train, y_train, X_test, y_test):
    """Trains final model on full training set and evaluates on test set."""
    params["use_label_encoder"] = False
    params["eval_metric"] = "logloss"

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    # Output metrics
    print("\n--- Final Test Set Classification Report ---")
    print(classification_report(y_test, y_pred))
    print(f"Final Test PR-AUC Score: {average_precision_score(y_test, y_probs):.3f}")

    return model


def plot_importance(model):
    """Plots feature importance using a controlled matplotlib figure."""
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(model, importance_type="weight", ax=ax, height=0.5)
    plt.title("XGBoost Feature Importance", fontsize=14)
    plt.tight_layout()
    plt.show()


def find_optimal_threshold(model, X_test, y_test):
    """Determines the probability threshold that maximizes the F1-Score."""
    y_probs = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

    # Calculate F1 score for every possible threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]

    print(f"Best F1: {f1_scores[best_idx]:.4f}")
    print(f"Best Threshold: {best_thresh:.4f}")

    return best_thresh


def train_final_model(params, X, y):
    """Trains model on provided data without evaluation."""
    train_params = params.copy()
    train_params["use_label_encoder"] = False
    train_params["eval_metric"] = "logloss"

    model = xgb.XGBClassifier(**train_params)
    model.fit(X, y)
    return model


# Pipeline Execution

if __name__ == "__main__":
    # 1. Data Ingestion & Labeling
    # Load dataset csv, scan audio, inject missing positives, and filter candidates
    dataset = build_dataset(CONF)

    # 2. Feature Extraction
    # Compute DSP features (spectral, rhythmic, RMS) for every candidate
    dataset_proc = features_batch(dataset)

    # 3. Data Splitting
    # Split train/test by track to prevent data leakage (same track cannot be in both sets)
    (X_train, y_train), (X_test, y_test), (X, y) = split(dataset_proc)

    # 4. Hyperparameter Tuning
    # Run Bayesian optimization using Optuna to find best XGBoost parameters
    best_params = run_optimization(X_train, y_train, n_trials=30)

    # 5. Model Evaluation
    # Train on X_train using best params, evaluate PR-AUC on X_test
    model = train_and_evaluate(best_params, X_train, y_train, X_test, y_test)
    plot_importance(model)

    # 6. Threshold Optimization
    # Find the probability threshold that maximizes F1-Score on the test set
    best_threshold = find_optimal_threshold(model, X_test, y_test)

    # 7. Final Training & Serialization
    # Retrain on the entire dataset (Train + Test) for the production model
    final_model = train_final_model(best_params, X, y)

    # 8. Save final model
    # Set model metadata and dump the model along with its metadata
    feature_names = list(X_train.columns)
    model_with_metadata = {
        "model": final_model,  # Trained XGBoost object
        "threshold": best_threshold,  # Optimized F1 threshold (0.6532)
        "features": feature_names,  # Features: ['rms_diff_short', 'bass_ratio', ...]
        "config": CONF,  # Settings: {'sr': 22050, 'snap_window': 2.0, ...}
        "meta": {
            "librosa_version": librosa.__version__,
            "xgb_version": xgb.__version__,
        },
    }

    joblib.dump(model_with_metadata, "model.joblib")
    print("Model saved successfully.")
