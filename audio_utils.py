from math import gcd
import numpy as np


def resample_audio(audio_array, orig_sr, target_sr=16000):
    """Resample 1-D mono float32 PCM audio from ``orig_sr`` to ``target_sr``."""
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(f"Sample rates must be positive (got {orig_sr} -> {target_sr})")
    arr = np.asarray(audio_array, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D mono audio array, got shape {arr.shape}")
    if arr.size == 0 or orig_sr == target_sr:
        return arr

    try:
        from scipy.signal import resample_poly

        g = gcd(int(orig_sr), int(target_sr))
        return resample_poly(arr, int(target_sr) // g, int(orig_sr) // g).astype(np.float32)
    except ImportError:
        pass

    # Pure-NumPy band-limited sinc / linear fallback when scipy is absent
    if target_sr == 2 * orig_sr:
        out = np.zeros(arr.size * 2, dtype=np.float32)
        out[0::2] = arr
        taps = np.sinc(np.arange(-15, 16, dtype=np.float32) * 0.5) * np.blackman(31).astype(np.float32)
        taps /= np.sum(taps) * 0.5
        return np.convolve(out, taps, mode="same").astype(np.float32)

    duration = arr.size / float(orig_sr)
    target_len = max(1, int(round(duration * target_sr)))
    src_idx = np.linspace(0.0, arr.size - 1, target_len, dtype=np.float32)
    return np.interp(src_idx, np.arange(arr.size, dtype=np.float32), arr).astype(np.float32)


def truncate_audio_to_last_n_seconds(audio_array, n_seconds=8, sample_rate=16000):
    """Truncate audio to last n seconds or pad with zeros to meet n seconds."""
    max_samples = n_seconds * sample_rate
    if len(audio_array) > max_samples:
        return audio_array[-max_samples:]
    elif len(audio_array) < max_samples:
        # Pad with zeros at the beginning
        padding = max_samples - len(audio_array)
        return np.pad(audio_array, (padding, 0), mode='constant', constant_values=0)
    return audio_array