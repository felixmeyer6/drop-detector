import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simpleaudio as sa
from pydub import AudioSegment
from pydub.generators import Sine

# Ensure local modules can be imported
sys.path.append(str(Path(__file__).parent))

from processors import (
    find_candidates,
)

warnings.filterwarnings("ignore")


# Init
CSV_PATH = Path("./dataset_train_raw.csv")
MUSIC_PATH = Path("./dataset_train")

CONF = {
    "snippet_dur": 5.0,  # Length of audio snippet played
    "pre_drop": 1.0,  # Seconds played before the drop
    "viz_win": 4.0,  # Duration of zoomed plot window
    "viz_resol": 8000,  # Resolution of the displayed envelopes
    "sr": 22050,  # Audio sample rate for analysis
    "min_distance": 2.0,  # Min seconds between candidates
    "score_lookback": 30,  # Frames for score calculation
    "snap_window": 2.0,  # Window for beat snapping
    "intro_s": 20,  # No drops in the first 20s
    "outro_s": 20,  # No drops in the last 20s
}


# Visualization
def get_viz_data(
    y: np.ndarray,
    sr: int,
    points: int = CONF["viz_resol"],
) -> Tuple[np.ndarray, np.ndarray]:
    if len(y) == 0:
        return np.zeros(1), np.zeros(1)
    chunk = max(1, int(np.ceil(len(y) / points)))
    pad_len = int(np.ceil(len(y) / chunk)) * chunk - len(y)
    y_padded = np.pad(y, (0, pad_len))
    env = np.abs(y_padded).reshape(-1, chunk).mean(axis=1)
    times = (np.arange(len(env)) * chunk + chunk / 2) / sr
    return times, env / (env.max() or 1)


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


# Main
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
            and self.df[(self.df.filename == fname) & (self.df.finished)].any().any()
        ):
            return

        confirmed_drops, rejected_drops = self.get_history(fname)
        reviewed_times = confirmed_drops + rejected_drops

        print(
            f"\n=== {fname} (Hist: {len(confirmed_drops)} yes, {len(rejected_drops)} no) ==="
        )

        # Generate all candidates
        candidates = find_candidates(fpath, CONF)

        if not candidates:
            print("No candidates found (filtered or low energy).")
            return

        # Load audio for playback
        try:
            full_audio = AudioSegment.from_file(fpath)
        except Exception as e:
            print(f"AudioSegment load error: {e}")
            return

        # Setup Visualization
        y = candidates[0]["y_ref"]
        sr = CONF["sr"]

        viz_t, viz_env = get_viz_data(y, sr)
        viz = DropVisualizer(
            viz_t, viz_env, len(y) / sr, f"{fname}", confirmed_drops=confirmed_drops
        )

        print(f"Reviewing {len(candidates)} candidates sorted by score...")

        # Review Loop
        for cand in candidates:
            t = cand["time"]

            if self.is_already_reviewed(t, reviewed_times):
                continue

            start_ms = max(0, (t - CONF["pre_drop"]) * 1000)
            end_ms = start_ms + (CONF["snippet_dur"] * 1000)

            print(
                f"Cand: {t:.2f}s (Score: {cand['score']:.2f})... ",
                end="",
                flush=True,
            )

            viz.update(
                t,
                (start_ms / 1000, end_ms / 1000),
                (t - CONF["viz_win"] / 2, t + CONF["viz_win"] / 2),
            )

            play_obj = self.play_snippet(full_audio[start_ms:end_ms], t, start_ms)

            # Interaction Loop
            action_taken = False
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
                    action_taken = True
                    break

            if action_taken and choice == "f":
                break

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
