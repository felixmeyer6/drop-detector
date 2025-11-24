import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd

try:
    import mutagen
    from mutagen.id3 import ID3, TXXX
    from mutagen.mp4 import MP4
except ImportError:
    mutagen = None

# Ensure local modules can be imported
sys.path.append(str(Path(__file__).parent))

from processors import (
    features_batch,
    generate_candidates,
)


def is_corrupt(file_path):
    """
    Runs ffmpeg on the file to check for errors.
    Returns (True, error_message) if corrupt or ffmpeg missing, (False, "") if clean.
    """
    try:
        command = ["ffmpeg", "-v", "error", "-i", str(file_path), "-f", "null", "-"]
        result = subprocess.run(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
        if result.stderr:
            return True, result.stderr.strip()
        return False, ""
    except FileNotFoundError:
        return True, "ffmpeg is not installed"
    except Exception as e:
        return True, str(e)


def write_drop_tag(fpath, times_list):
    """
    Writes the detected drop times to the file metadata.
    Handles ID3 (MP3), MP4 (M4A), and Vorbis (FLAC/OGG).
    """
    if mutagen is None:
        print("Warning: 'mutagen' library not found. Skipping tagging.")
        return

    times_str = ",".join([f"{t:.2f}" for t in times_list])

    try:
        audio = mutagen.File(fpath)
        if audio is None:
            return

        # Handle MP3, AIFF
        if hasattr(audio, "tags") and isinstance(
            audio.tags, (mutagen.id3.ID3, mutagen.id3.ID3NoHeader)
        ):
            audio.tags.add(TXXX(encoding=3, desc="DROP_TIME", text=times_str))

        # Handle MP4, M4A
        elif isinstance(audio, MP4):
            audio.tags["----:com.apple.iTunes:DROP_TIME"] = times_str.encode("utf-8")

        # Handle FLAC, OGG
        else:
            try:
                audio["DROP_TIME"] = times_str
            except Exception:
                pass

        audio.save()

    except Exception as e:
        print(f"Error tagging {Path(fpath).name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Detect musical drops in audio files.")

    # Inputs
    parser.add_argument(
        "--folder",
        "-f",
        required=True,
        type=Path,
        help="Path to the directory containing the audio files.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=Path,
        default=Path(__file__).parent / "model.joblib",
        help="Path to the trained model file.",
    )

    # Cutoff
    parser.add_argument(
        "--threshold",
        "-T",
        type=float,
        default=None,
        help="Manual confidence threshold (0.0 - 1.0). Overrides model default.",
    )

    parser.add_argument(
        "--topk",
        "-k",
        type=int,
        default=None,
        help="Output the top K drops per track. If set without a threshold, ignores confidence scores.",
    )

    # Outputs
    default_csv = Path(__file__).parent / "model_predictions.csv"
    parser.add_argument(
        "--csv",
        "-c",
        nargs="?",
        const=default_csv,
        default=None,
        type=Path,
        help="Save predictions to CSV. Default dir is where the script is.",
    )

    parser.add_argument(
        "--tag",
        "-t",
        action="store_true",
        help="Write drop times to audio file metadata (Tag: DROP_TIME).",
    )

    args = parser.parse_args()

    # 0. Load Model
    if not args.model.exists():
        print(f"Error: Model file not found at {args.model}")
        sys.exit(1)

    try:
        model_data = joblib.load(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    clf = model_data.get("model")
    model_optimal_threshold = model_data.get("threshold", 0.5)
    conf = model_data.get("config", {})
    required_features = model_data.get("features", [])
    meta = model_data.get("meta", {})

    # 1. Set Cutoff
    cutoff_value = 0.5
    logic_msg = ""

    if args.threshold is not None:
        # User Manual Threshold
        cutoff_value = args.threshold
        logic_msg = f"User Override: {cutoff_value:.4f}"

    elif args.topk is not None:
        # or Top-K Mode
        cutoff_value = 0.0
        logic_msg = "Top-K Mode: Threshold disabled (0.0)"

    else:
        # or Model Default
        cutoff_value = model_optimal_threshold
        logic_msg = f"Model Optimal: {cutoff_value:.4f}"

    print("\nModel Loaded.")
    print("-" * 30)
    print(f"Threshold Logic: {logic_msg}")
    print(f"Config:    {conf}")
    print(f"Meta:      {meta}")
    print("-" * 30)
    print(f"Features ({len(required_features)}):")
    print(required_features)
    print("-" * 30)

    # 2. Scan Folder
    if not args.folder.exists():
        print(f"Error: Folder not found at {args.folder}")
        sys.exit(1)

    print(f"\nScanning {args.folder} for audio files...")

    AUDIO_EXTENSIONS = {".wav", ".mp3", ".aiff", ".flac", ".m4a", ".ogg"}
    found_files = []
    for fpath in args.folder.rglob("*"):
        if fpath.is_file() and fpath.suffix.lower() in AUDIO_EXTENSIONS:
            found_files.append(fpath)

    if not found_files:
        print("No audio files found.")
        sys.exit(0)

    print(f"Found {len(found_files)} potential audio files.")

    # 3. Integrity Checks
    valid_files = []
    print("Checking file integrity...")

    for fpath in found_files:
        is_bad, error_msg = is_corrupt(fpath)
        if is_bad:
            continue
        valid_files.append(fpath)

    print(f"Proceeding with {len(valid_files)} valid files.")

    if not valid_files:
        sys.exit(0)

    # 4. Compute Candidates
    print("\nGenerating candidates...")
    all_candidates: List[Dict] = []
    files_with_candidates = set()

    for fpath in valid_files:
        candidates = generate_candidates(fpath, conf)

        # --- LOGGING START ---
        if candidates:
            # Extract times for logging
            times_log = [f"{c['time']:.2f}s" for c in candidates]
            print(f"  > {fpath.name}: {len(candidates)} candidates -> {times_log}")
            files_with_candidates.add(str(fpath))
        else:
            print(f"  > {fpath.name}: No candidates found.")
        # --- LOGGING END ---

        for cand in candidates:
            cand["fpath"] = str(fpath)
            # Optimization: clear memory
            cand.pop("y_ref", None)
            cand.pop("env", None)
            cand.pop("fps", None)
            all_candidates.append(cand)

    if not all_candidates:
        print(
            "No drop candidates detected in any files (Audio might be too quiet/clean)."
        )
        sys.exit(0)

    df_candidates = pd.DataFrame(all_candidates)
    print(
        f"Generated {len(df_candidates)} candidates across {df_candidates['fpath'].nunique()} files."
    )

    # 5. Compute Features
    df_features = features_batch(df_candidates)

    # 6. Run Inference & Filter
    print("Running inference...")

    try:
        X = df_features[required_features]
    except KeyError as e:
        print(
            f"Error: Generated features do not match model requirements. Missing: {e}"
        )
        sys.exit(1)

    probs = clf.predict_proba(X)[:, 1]
    df_features["confidence"] = probs

    # Apply Threshold (0 if topk is set)
    df_features["is_drop"] = probs >= cutoff_value
    df_drops = df_features[df_features["is_drop"]].copy()

    # Sort by File -> Confidence (Descending)
    df_drops.sort_values(
        by=["fpath", "confidence"], ascending=[True, False], inplace=True
    )

    # Apply Top-K (if requested)
    if args.topk is not None:
        print(f"Filtering top {args.topk} drops per track...")
        df_drops = df_drops.groupby("fpath").head(args.topk)

    output_df = df_drops[["fpath", "time", "confidence"]].copy()

    # 7. Diagnostics
    files_with_drops = set(output_df["fpath"].unique())
    files_scanned_set = set(str(f) for f in valid_files)

    print("\n")
    print(
        f"Detected {len(output_df)} drops across {len(files_with_drops)}/{len(files_scanned_set)} files."
    )

    missing_files = files_scanned_set - files_with_drops
    if missing_files:
        print("\nFiles with no drops detected:")
        for fpath_str in sorted(list(missing_files)):
            if fpath_str not in files_with_candidates:
                print(
                    f"  - {Path(fpath_str).name}: No signal candidates found (Audio too quiet?)"
                )
            else:
                file_feats = df_features[df_features["fpath"] == fpath_str]
                max_conf = (
                    file_feats["confidence"].max() if not file_feats.empty else 0.0
                )
                print(
                    f"  X {Path(fpath_str).name}: Best confidence {max_conf:.4f} < Cutoff {cutoff_value:.4f}"
                )
        print("-" * 30)

    # 8. Output (CSV & Tag)
    if args.csv is not None:
        try:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            output_df.to_csv(args.csv, index=False)
            print(f"\nSaved predictions to: {args.csv}")
        except Exception as e:
            print(f"\nError saving CSV: {e}")

    if args.tag:
        if output_df.empty:
            print("\nNo drops to tag.")
        else:
            print("\nWriting metadata tags...")
            grouped = output_df.groupby("fpath")
            for fpath_str, group in grouped:
                times = sorted(group["time"].tolist())
                write_drop_tag(fpath_str, times)
            print("Tagging complete.")


if __name__ == "__main__":
    main()
