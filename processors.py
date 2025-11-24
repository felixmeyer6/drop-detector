import sys
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# Ensure local modules can be imported
sys.path.append(str(Path(__file__).parent))

from utilities import (
    bandpass,
    bass_ratio,
    calculate_rms,
    compute_envelope,
    low_shelf,
    pulse_clarity,
    snap_time,
)


def get_score_at_time(t_sec: float, env: np.ndarray, fps: float, CONF: Dict) -> float:
    """Calculates a 'drop score' based on the inverse of the preceding average energy."""
    idx = int(t_sec * fps)
    if idx >= len(env):
        return 0.0
    score_start = max(0, idx - int(CONF["score_lookback"]))
    if idx <= score_start:
        return 0.0
    local_mean = np.mean(env[score_start:idx])
    return 1.0 - local_mean


def generate_candidates(fpath: Path, CONF: Dict) -> List[Dict]:
    """Scans audio for candidate drops by finding envelope threshold crossings."""
    # Load and Proprocess Audio
    try:
        y, sr = librosa.load(fpath, sr=CONF["sr"])
        track_duration = len(y) / sr

        if y is None or len(y) < CONF["sr"] * 1.0:
            return []
        if not np.isfinite(y).all():
            return []
    except Exception:
        return []

    y_boosted = low_shelf(y, sr, cutoff=600, gain_db=6.0)
    env, fps = compute_envelope(y_boosted, sr)

    raw_candidates = []

    # Iterative vars
    curr_threshold = 0.85
    min_threshold = 0.40
    step = 0.10

    # Iterative Search
    while curr_threshold >= min_threshold:
        is_above = env >= curr_threshold
        # Find indices where envelope crosses threshold from below
        crossings = np.where(np.diff(is_above.astype(int)) == 1)[0] + 1

        for idx in crossings:
            t_sec = idx / fps

            # Filter intros/outros
            if t_sec < CONF["intro_s"] or t_sec > (track_duration - CONF["outro_s"]):
                continue

            # Gap Check
            gap_start = max(0, idx - int(2.0 * fps))
            local_mean = np.mean(env[gap_start:idx])

            # Skip if mean volume before is too high relative to threshold
            if local_mean > (curr_threshold * 0.85):
                continue

            # Scoring
            score_start = max(0, idx - int(CONF["score_lookback"] * fps))
            score = 1.0 - np.mean(env[score_start:idx])

            raw_candidates.append(
                {
                    "raw_time": t_sec,
                    "score": score,
                    "y_ref": y,
                    "src_thresh": curr_threshold,
                }
            )

        curr_threshold -= step

    # Sort by score descending
    raw_candidates.sort(key=lambda x: x["score"], reverse=True)

    final_candidates = []

    for cand in raw_candidates:
        # Snap to beat grid
        t_snapped = snap_time(cand["raw_time"], cand["y_ref"], CONF)

        # Deduplicate based on time distance
        if any(
            abs(t_snapped - existing["time"]) < CONF["min_distance"]
            for existing in final_candidates
        ):
            continue

        final_candidates.append(
            {
                "time": t_snapped,
                "score": cand["score"],
                "y_ref": y,
                "env": env,
                "fps": fps,
            }
        )

    return final_candidates


def features_file(fpath, group_df, sr=22050):
    """
    Extracts contextual audio features for a list of candidate timestamps.

    Strategy:
    Compares the audio properties *after* the candidate time (impact) against
    multiple windows *before* the candidate time (buildup/breakdown).
    """
    results = []

    try:
        # Load Audio
        y_raw, _ = librosa.load(fpath, sr=sr)
        total_duration = len(y_raw) / sr

        # Pre-Compute Signals
        y = low_shelf(y_raw, sr)

        y_bass = bandpass(y_raw, sr, low_cut=50, high_cut=150)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        # Compute Beat Grid
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        def get_slice_samples(src_arr, t_start, t_end):
            s = max(0, int(t_start * sr))
            e = min(len(src_arr), int(t_end * sr))
            return src_arr[s:e]

        def get_slice_frames(src_arr, t_start, t_end, hop_length=512):
            s = max(0, int(t_start * sr / hop_length))
            e = min(len(src_arr), int(t_end * sr / hop_length))
            return src_arr[s:e]

        # Iterate Candidates
        for idx, row in group_df.iterrows():
            t = row["time"]

            # RMS Features
            # Does the volume increase and sustain?
            def get_rms(t_s, t_e):
                return calculate_rms(get_slice_samples(y, t_s, t_e))

            rms_post_short = get_rms(t + 0.5, t + 4.0)

            feat_rms_short_diff_far = rms_post_short - get_rms(t - 20, t - 5)
            feat_rms_short_diff_mid = rms_post_short - get_rms(t - 5, t - 2)
            feat_rms_short_diff_near = rms_post_short - get_rms(t - 2, t - 0.5)

            rms_post_med = get_rms(t + 0.5, t + 10.0)

            feat_rms_med_diff_far = rms_post_med - get_rms(t - 20, t - 5)
            feat_rms_med_diff_mid = rms_post_med - get_rms(t - 5, t - 2)
            feat_rms_med_diff_near = rms_post_med - get_rms(t - 2, t - 0.5)

            rms_post_long = get_rms(t + 0.5, t + 15.0)

            feat_rms_long_diff_far = rms_post_long - get_rms(t - 30, t - 10)
            feat_rms_long_diff_mid = rms_post_long - get_rms(t - 10, t - 5)
            feat_rms_long_diff_near = rms_post_long - get_rms(t - 5, t - 1.0)

            # Pulse Clarity Features
            # Is there a defined rythm?
            pc_post_far = pulse_clarity(get_slice_samples(y, t, t + 30), sr)
            pc_pre_far = pulse_clarity(get_slice_samples(y, t - 30, t - 5), sr)
            feat_pc_diff_far = pc_post_far - pc_pre_far
            pc_post_mid = pulse_clarity(get_slice_samples(y, t, t + 12), sr)
            pc_pre_mid = pulse_clarity(get_slice_samples(y, t - 20, t - 5), sr)
            feat_pc_diff_mid = pc_post_mid - pc_pre_mid
            pc_post_near = pulse_clarity(get_slice_samples(y, t, t + 8), sr)
            pc_pre_near = pulse_clarity(get_slice_samples(y, t - 10, t - 3), sr)
            feat_pc_diff_near = pc_post_near - pc_pre_near

            # Bass Ratio Features
            # Does the bass increase?
            br_post_far = bass_ratio(get_slice_samples(y, t + 0.5, t + 10), sr)
            br_pre_far = bass_ratio(get_slice_samples(y, t - 5, t - 2), sr)
            feat_bass_ratio_diff_far = br_post_far - br_pre_far
            br_post_mid = bass_ratio(get_slice_samples(y, t + 0.3, t + 7), sr)
            br_pre_mid = bass_ratio(get_slice_samples(y, t - 5, t - 1), sr)
            feat_bass_ratio_diff_mid = br_post_mid - br_pre_mid
            br_post_near = bass_ratio(get_slice_samples(y, t + 0.1, t + 5), sr)
            br_pre_near = bass_ratio(get_slice_samples(y, t - 3, t - 0.5), sr)
            feat_bass_ratio_diff_near = br_post_near - br_pre_near

            # Grid Alignment Features
            # Is the candidate on a 4,8,16,32 bar section change?
            if len(beat_times) > 0:
                closest_beat_idx = (np.abs(beat_times - t)).argmin()
                grid_4 = np.cos(2 * np.pi * closest_beat_idx / 16)
                grid_8 = np.cos(2 * np.pi * closest_beat_idx / 32)
                grid_16 = np.cos(2 * np.pi * closest_beat_idx / 64)
                grid_32 = np.cos(2 * np.pi * closest_beat_idx / 128)
            else:
                grid_4, grid_8, grid_16, grid_32 = 0, 0, 0, 0

            # Transient Dominance Ratio Features
            # Is the hit significant compared to its surroundings?
            oe_local = get_slice_frames(onset_env, t - 0.1, t + 0.1)
            max_local = np.max(oe_local) if len(oe_local) > 0 else 0.0

            def calc_transient_dom(ctx_start, ctx_end):
                oe_ctx = get_slice_frames(onset_env, t + ctx_start, t + ctx_end)
                max_ctx = np.max(oe_ctx) if len(oe_ctx) > 0 else 0.0
                return max_local / (max_ctx + 1e-6)

            feat_td_short = calc_transient_dom(-0.5, 2.0)
            feat_td_med = calc_transient_dom(-0.5, 10.0)
            feat_td_long = calc_transient_dom(-2.0, 20.0)

            # Bass Continuity Features
            # Does the bass sustain?
            y_bass_impact = get_slice_samples(y_bass, t, t + 0.30)
            rms_bass_impact = calculate_rms(y_bass_impact)

            def calc_bass_cont(sus_start, sus_end):
                if rms_bass_impact < 1e-6:
                    return 0.0
                y_sus = get_slice_samples(y_bass, t + sus_start, t + sus_end)
                return calculate_rms(y_sus) / rms_bass_impact

            feat_bc_tight = calc_bass_cont(0.25, 0.5)
            feat_bc_std = calc_bass_cont(0.5, 1.0)
            feat_bc_loose = calc_bass_cont(1.0, 2.0)

            # Future Energy Dominance Features
            # Is there a louder section coming up later?
            y_curr_1s = get_slice_samples(y, t, t + 1.0)
            rms_curr_1s = calculate_rms(y_curr_1s)

            def calc_future_dom(fut_start, fut_end):
                y_fut = get_slice_samples(y, t + fut_start, t + fut_end)
                if len(y_fut) == 0:
                    return 0.0
                n_chunks = max(1, int(len(y_fut) / sr))
                chunks = np.array_split(y_fut, n_chunks)
                rms_fut_max = max([calculate_rms(c) for c in chunks])
                return rms_curr_1s / (rms_fut_max + 1e-6)

            feat_fed_near = calc_future_dom(1.0, 5.0)
            feat_fed_med = calc_future_dom(2.0, 10.0)
            feat_fed_far = calc_future_dom(5.0, 20.0)
            feat_fed_super_far = calc_future_dom(5.0, 40.0)

            # Positional Features
            # Where is the candidate in the track?
            feat_norm_time = t / total_duration if total_duration > 0 else 0

            results.append(
                {
                    "index": idx,
                    "rms_diff_short_far": feat_rms_short_diff_far,
                    "rms_diff_short_mid": feat_rms_short_diff_mid,
                    "rms_diff_short_near": feat_rms_short_diff_near,
                    "rms_diff_med_far": feat_rms_med_diff_far,
                    "rms_diff_med_mid": feat_rms_med_diff_mid,
                    "rms_diff_med_near": feat_rms_med_diff_near,
                    "rms_diff_long_far": feat_rms_long_diff_far,
                    "rms_diff_long_mid": feat_rms_long_diff_mid,
                    "rms_diff_long_near": feat_rms_long_diff_near,
                    "pulse_clarity_diff_far": feat_pc_diff_far,
                    "pulse_clarity_diff_mid": feat_pc_diff_mid,
                    "pulse_clarity_diff_near": feat_pc_diff_near,
                    "bass_ratio_diff_far": feat_bass_ratio_diff_far,
                    "bass_ratio_diff_mid": feat_bass_ratio_diff_mid,
                    "bass_ratio_diff_near": feat_bass_ratio_diff_near,
                    "transient_dominance_short": feat_td_short,
                    "transient_dominance_med": feat_td_med,
                    "transient_dominance_long": feat_td_long,
                    "bass_continuity_tight": feat_bc_tight,
                    "bass_continuity_std": feat_bc_std,
                    "bass_continuity_loose": feat_bc_loose,
                    "future_energy_dominance_near": feat_fed_near,
                    "future_energy_dominance_med": feat_fed_med,
                    "future_energy_dominance_far": feat_fed_far,
                    "future_energy_dominance_super_far": feat_fed_super_far,
                    "grid_phrase_4bar": grid_4,
                    "grid_phrase_8bar": grid_8,
                    "grid_phrase_16bar": grid_16,
                    "grid_phrase_32bar": grid_32,
                    "norm_time": feat_norm_time,
                }
            )

    except Exception as e:
        print(f"Error processing {fpath}: {e}")
        for idx, row in group_df.iterrows():
            results.append({"index": idx})

    return results


def features_batch(df, n_jobs=-1):
    """Computes features for all tracks."""
    grouped = df.groupby("fpath")
    tasks = [(fpath, group) for fpath, group in grouped]
    print(f"Extracting features for {len(df)} candidates across {len(tasks)} files...")

    results_list = Parallel(n_jobs=n_jobs)(
        delayed(features_file)(fpath, group) for fpath, group in tqdm(tasks)
    )

    flat_results = [item for sublist in results_list for item in sublist]
    features_df = pd.DataFrame(flat_results).set_index("index")
    return df.join(features_df)
