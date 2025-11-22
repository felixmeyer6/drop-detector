import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
import simpleaudio as sa
from pydub import AudioSegment
from pydub.generators import Sine

warnings.filterwarnings("ignore")


# ================= CONFIGURATION =================
CSV_PATH = Path("./training_dataset.csv")
MUSIC_PATH = Path("./training_dataset")

CONF = {
    "snippet_dur": 5.0,  # Length of audio snippet played
    "pre_drop": 1.0,  # Seconds played before the drop
    "suppress_win": 1.0,  # Radius to exclude nearby duplicates
    "viz_win": 4.0,  # Duration of zoomed plot window
    "sr": 22050,  # Audio sample rate for analysis
    "start_threshold": 0.85,  # Initial detection strictness
    "min_threshold": 0.40,  # Minimum strictness before stopping
    "threshold_step": 0.10,  # Decrease in strictness per pass
}


# ================= SIGNAL PROCESSING =================
def apply_low_shelf(
    y: np.ndarray, sr: int, cutoff: float = 500.0, gain_db: float = 3.0
) -> np.ndarray:
    sos = scipy.signal.butter(4, cutoff, btype="low", fs=sr, output="sos")
    low = scipy.signal.sosfilt(sos, y)
    high = y - low
    return (low * (10 ** (gain_db / 20.0))) + high


def compute_envelope(
    y: np.ndarray, sr: int, hop_length: int = 512
) -> Tuple[np.ndarray, float]:
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    window = np.hanning(int(0.5 * sr / hop_length) | 1)
    env = scipy.signal.convolve(rms, window / window.sum(), mode="same")
    max_val = env.max()
    return (env / max_val if max_val > 1e-6 else env), sr / hop_length


def find_candidates(y: np.ndarray, sr: int, threshold: float) -> List[Dict]:
    y_boosted = apply_low_shelf(y, sr, cutoff=600, gain_db=6.0)
    env, fps = compute_envelope(y_boosted, sr)

    is_above = env >= threshold
    crossings = np.where(np.diff(is_above.astype(int)) == 1)[0] + 1

    candidates = []
    for idx in crossings:
        t_sec = idx / fps
        if t_sec < 2.0:
            continue

        gap_start = max(0, idx - int(2.0 * fps))
        local_mean = np.mean(env[gap_start:idx])

        if local_mean > (threshold * 0.85):
            continue

        score_start = max(0, idx - int(30.0 * fps))
        candidates.append({"time": t_sec, "score": 1.0 - np.mean(env[score_start:idx])})

    return sorted(candidates, key=lambda x: x["score"], reverse=True)


def snap_time(time: float, y: np.ndarray, sr: int) -> float:
    win_rad = 2.0
    s_start = max(0, int((time - win_rad) * sr))
    s_end = min(len(y), int((time + win_rad) * sr))

    if s_start >= s_end:
        return time

    y_slice = apply_low_shelf(y[s_start:s_end], sr, cutoff=500, gain_db=3.0)
    onsets = librosa.onset.onset_detect(
        y=y_slice, sr=sr, units="samples", backtrack=True
    )

    if len(onsets) == 0:
        return time

    best_t, max_rise = time, -np.inf
    pre, post = int(0.5 * sr), int(0.5 * sr)

    for onset in onsets:
        if onset - pre < 0 or onset + post >= len(y_slice):
            continue
        rise = np.sqrt(np.mean(y_slice[onset : onset + post] ** 2)) - np.sqrt(
            np.mean(y_slice[onset - pre : onset] ** 2)
        )
        if rise > max_rise:
            max_rise = rise
            best_t = (s_start + onset) / sr

    return best_t


def get_viz_data(
    y: np.ndarray,
    sr: int,
    points: int = 8000,  # Resolution of the displayed envelopes
) -> Tuple[np.ndarray, np.ndarray]:
    if len(y) == 0:
        return np.zeros(1), np.zeros(1)
    chunk = max(1, int(np.ceil(len(y) / points)))
    pad_len = int(np.ceil(len(y) / chunk)) * chunk - len(y)
    y_padded = np.pad(y, (0, pad_len))
    env = np.abs(y_padded).reshape(-1, chunk).mean(axis=1)
    times = (np.arange(len(env)) * chunk + chunk / 2) / sr
    return times, env / (env.max() or 1)


# ================= VISUALIZATION =================
class DropVisualizer:
    def __init__(
        self,
        times: np.ndarray,
        env: np.ndarray,
        duration: float,
        title: str,
        confirmed_drops: List[float] = None,
    ):
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.canvas.manager.set_window_title("Drop Detector")
        self.duration = duration

        # Full View
        self.ax1.fill_between(times, -env, env, color="k")
        self.l1 = self.ax1.axvline(0, color="r", ls="--")
        self.ax1.set_xlim(0, duration)
        self.ax1.set_ylim(-1, 1)
        self.ax1.set_title(title)

        # Zoom View
        self.ax2.fill_between(times, -env, env, color="k")
        self.l2 = self.ax2.axvline(0, color="r", ls="--", lw=2)
        self.ax2.set_ylim(-1, 1)

        self.zones = []

        if confirmed_drops:
            for t in confirmed_drops:
                self.add_marker(t)

    def add_marker(self, t: float):
        self.ax1.axvline(t, color="r", alpha=0.4, ls="-", lw=1)
        self.ax2.axvline(t, color="r", alpha=0.4, ls="-", lw=1)

    def update(
        self, t: float, snip_rng: Tuple[float, float], bound_rng: Tuple[float, float]
    ):
        for z in self.zones:
            z.remove()
        self.zones.clear()

        self.l1.set_xdata([t, t])
        self.l2.set_xdata([t, t])

        for ax in (self.ax1, self.ax2):
            self.zones.append(ax.axvspan(*snip_rng, color="r", alpha=0.15))
            self.zones.append(ax.axvspan(*bound_rng, color="b", alpha=0.15))

        self.ax2.set_xlim(max(0, t - 7.5), min(self.duration, t + 7.5))
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

    def close(self):
        plt.close(self.fig)


# ================= MAIN =================
class DropAssistant:
    def __init__(self):
        self.df = self._load_db()

    def _load_db(self) -> pd.DataFrame:
        if CSV_PATH.exists():
            df = pd.read_csv(CSV_PATH)
            if "finished" not in df.columns:
                df["finished"] = False
            return df
        return pd.DataFrame(columns=["filename", "timestamp", "label", "finished"])

    def save_entry(self, fname: str, time: float, label: int):
        new_row = pd.DataFrame(
            {
                "filename": [fname],
                "timestamp": [round(time, 2)],
                "label": [label],
                "finished": [False],
            }
        )
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.df.to_csv(CSV_PATH, index=False)

    def mark_finished(self, fname: str):
        """Marks all rows for this file as finished. Creates a placeholder if needed."""
        if len(self.df[self.df.filename == fname]) == 0:
            self.save_entry(fname, 0.0, -1)

        self.df.loc[self.df.filename == fname, "finished"] = True
        self.df.to_csv(CSV_PATH, index=False)

    def get_history(self, fname: str) -> Tuple[List[float], List[float]]:
        subset = self.df[self.df.filename == fname]
        confirmed = subset[subset.label == 1].timestamp.tolist()
        rejected = subset[subset.label == 0].timestamp.tolist()
        return confirmed, rejected

    def is_already_reviewed(self, t: float, history: List[float]) -> bool:
        return any(abs(t - h) < 1.0 for h in history)

    def play_snippet(self, audio: AudioSegment, drop_t: float, start_ms: int):
        beep = Sine(1000).to_audio_segment(duration=100).apply_gain(-5)
        snippet = audio.overlay(beep, position=(drop_t * 1000) - start_ms)
        return sa.play_buffer(
            snippet.raw_data, snippet.channels, snippet.sample_width, snippet.frame_rate
        )

    def _process_file(self, fpath: Path):
        fname = fpath.name

        # Skip if finished
        if (
            not self.df.empty
            and "finished" in self.df.columns
            and self.df[(self.df.filename == fname) & (self.df.finished == True)]
            .any()
            .any()
        ):
            return

        confirmed_drops, rejected_drops = self.get_history(fname)
        reviewed_times = confirmed_drops + rejected_drops

        print(
            f"\n=== {fname} (Hist: {len(confirmed_drops)} yes, {len(rejected_drops)} no) ==="
        )

        try:
            y, sr = librosa.load(fpath, sr=CONF["sr"])
            full_audio = AudioSegment.from_file(fpath)
        except Exception as e:
            print(f"Load error: {e}")
            return

        viz_t, viz_env = get_viz_data(y, sr)
        viz = DropVisualizer(
            viz_t, viz_env, len(y) / sr, f"{fname}", confirmed_drops=confirmed_drops
        )

        curr_threshold = CONF["start_threshold"]
        finished_file = False

        while curr_threshold >= CONF["min_threshold"] and not finished_file:
            raw_cands = find_candidates(y, sr, threshold=curr_threshold)
            valid_candidates = []

            for c in raw_cands:
                t = c["time"]

                if self.is_already_reviewed(t, reviewed_times):
                    continue

                t_snap = snap_time(t, y, sr)

                if self.is_already_reviewed(t_snap, reviewed_times):
                    continue

                if any(abs(t_snap - v["time"]) < 1.0 for v in valid_candidates):
                    continue

                c["time"] = t_snap
                valid_candidates.append(c)

            if not valid_candidates:
                print(
                    f"No new candidates at threshold {curr_threshold:.2f}. Widening search..."
                )
                curr_threshold -= CONF["threshold_step"]
                continue

            print(
                f"Found {len(valid_candidates)} new candidates at threshold {curr_threshold:.2f}."
            )

            for cand in valid_candidates:
                t = cand["time"]

                if self.is_already_reviewed(t, reviewed_times):
                    continue

                start_ms = max(0, (t - CONF["pre_drop"]) * 1000)
                end_ms = start_ms + (CONF["snippet_dur"] * 1000)

                print(
                    f"[{curr_threshold:.2f}] Cand: {t:.2f}s (Score: {cand['score']:.2f})... ",
                    end="",
                    flush=True,
                )
                viz.update(
                    t,
                    (start_ms / 1000, end_ms / 1000),
                    (t - CONF["viz_win"] / 2, t + CONF["viz_win"] / 2),
                )
                play_obj = self.play_snippet(full_audio[start_ms:end_ms], t, start_ms)

                while True:
                    choice = input("[y]es, [n]o, [r]eplay, [f]inish: ").lower().strip()
                    if play_obj.is_playing():
                        play_obj.stop()

                    if choice == "r":
                        play_obj = self.play_snippet(
                            full_audio[start_ms:end_ms], t, start_ms
                        )
                    elif choice == "y":
                        self.save_entry(fname, t, 1)
                        viz.add_marker(t)
                        reviewed_times.append(t)
                        print("Confirmed.")
                        break
                    elif choice == "n":
                        self.save_entry(fname, t, 0)
                        reviewed_times.append(t)
                        print("Rejected.")
                        break
                    elif choice == "f":
                        self.mark_finished(fname)
                        finished_file = True
                        break

                if finished_file:
                    break

            if not finished_file:
                curr_threshold -= CONF["threshold_step"]

        viz.close()

    def run(self):
        extensions = {".mp3", ".wav", ".flac", ".aiff", ".m4a"}
        files = sorted(
            [p for p in MUSIC_PATH.rglob("*") if p.suffix.lower() in extensions]
        )
        print(f"Found {len(files)} files.")

        for f in files:
            self._process_file(f)
        print("\nDone.")


if __name__ == "__main__":
    DropAssistant().run()
