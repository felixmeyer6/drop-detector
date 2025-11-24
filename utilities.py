from typing import Tuple

import librosa
import numpy as np
import scipy.signal


def low_shelf(
    y: np.ndarray, sr: int, cutoff: float = 500.0, gain_db: float = 3.0
) -> np.ndarray:
    """Boosts bass frequencies, making drops more prominent."""
    sos = scipy.signal.butter(4, cutoff, btype="low", fs=sr, output="sos")
    low = scipy.signal.sosfilt(sos, y)
    high = y - low
    return (low * (10 ** (gain_db / 20.0))) + high


def bandpass(y: np.ndarray, sr: int, low_cut=50, high_cut=150) -> np.ndarray:
    """Applies a butterworth bandpass filter to isolate a specific frequency range."""
    sos = scipy.signal.butter(4, [low_cut, high_cut], btype="band", fs=sr, output="sos")
    return scipy.signal.sosfilt(sos, y)


def compute_envelope(
    y: np.ndarray, sr: int, hop_length: int = 512
) -> Tuple[np.ndarray, float]:
    """Computes the normalized amplitude envelope of the signal."""
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    window = np.hanning(int(0.5 * sr / hop_length) | 1)
    env = scipy.signal.convolve(rms, window / window.sum(), mode="same")
    max_val = env.max()
    fps = sr / hop_length
    return (env / max_val if max_val > 1e-6 else env), fps


def calculate_rms(y_slice):
    """Calculates the Root Mean Square (RMS) amplitude."""
    if len(y_slice) == 0:
        return 0.0
    return np.sqrt(np.mean(y_slice**2))


def bass_ratio(y: np.ndarray, sr: int) -> float:
    """Calculates the ratio of low-frequency energy to total energy."""
    if len(y) == 0:
        return 0.0
    S = np.abs(librosa.stft(y, n_fft=2048))
    energy_spectrum = np.sum(S, axis=1)
    total_energy = np.sum(energy_spectrum)
    if total_energy < 1e-6:
        return 0.0
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mask = (freqs >= 60) & (freqs <= 120)
    bass_energy = np.sum(energy_spectrum[mask])
    return bass_energy / total_energy


def pulse_clarity(y: np.ndarray, sr: int = 22050) -> float:
    """
    Estimates the strength of the beat by calculating the ratio of the highest autocorrelation
    peak to the mean of the autocorrelation signal. Higher values indicate a clearer rhythm.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if len(onset_env) == 0 or np.sum(onset_env) == 0:
        return 0.0
    ac = librosa.autocorrelate(onset_env, max_size=500)
    if len(ac) < 2:
        return 0.0
    ac_no_lag0 = ac[1:]
    max_peak = np.max(ac_no_lag0)
    mean_val = np.mean(ac_no_lag0)
    if mean_val < 1e-6:
        return 0.0
    return max_peak / mean_val


def snap_time(time: float, y: np.ndarray, CONF) -> float:
    """Refines a candidate timestamp by snapping it to the nearest transient."""
    win_rad = CONF["snap_window"]
    s_start = max(0, int((time - win_rad) * CONF["sr"]))
    s_end = min(len(y), int((time + win_rad) * CONF["sr"]))

    if s_start >= s_end:
        return time

    y_slice = low_shelf(y[s_start:s_end], CONF["sr"], cutoff=500, gain_db=3.0)

    onsets = librosa.onset.onset_detect(
        y=y_slice, sr=CONF["sr"], units="samples", backtrack=True
    )

    if len(onsets) == 0:
        return time

    best_t, max_rise = time, -np.inf
    pre, post = int(0.5 * CONF["sr"]), int(0.5 * CONF["sr"])

    for onset in onsets:
        if onset - pre < 0 or onset + post >= len(y_slice):
            continue

        rise = np.sqrt(np.mean(y_slice[onset : onset + post] ** 2)) - np.sqrt(
            np.mean(y_slice[onset - pre : onset] ** 2)
        )
        if rise > max_rise:
            max_rise = rise
            best_t = (s_start + onset) / CONF["sr"]

    return best_t
