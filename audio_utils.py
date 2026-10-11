import numpy as np
from scipy.signal import butter, resample_poly, sosfilt


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


TELEPHONY_CHANNELS = ("wideband", "narrowband", "pstn")


def simulate_telephony(audio_array, channel, sample_rate=16000):
    """Pass audio through a simulated phone line and return it at the original rate and length.

    narrowband: resample to 8 kHz, 8-bit G.711 mu-law, back to sample_rate.
    pstn: narrowband plus the 300-3400 Hz voice band filter of a phone line.
    """
    if channel not in TELEPHONY_CHANNELS:
        raise ValueError(f"unknown channel {channel!r}, expected one of {TELEPHONY_CHANNELS}")
    audio = np.asarray(audio_array, dtype=np.float32)
    if channel == "wideband":
        return audio

    x = resample_poly(audio.astype(np.float64), 8000, sample_rate)
    if channel == "pstn":
        x = sosfilt(butter(4, [300, 3400], btype="bandpass", fs=8000, output="sos"), x)

    mu = 255
    x = np.clip(x, -1.0, 1.0)
    codes = np.round((np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu) + 1) / 2 * mu)
    y = codes / mu * 2 - 1
    x = np.sign(y) * np.expm1(np.abs(y) * np.log1p(mu)) / mu

    out = resample_poly(x, sample_rate, 8000).astype(np.float32)
    if len(out) >= len(audio):
        return out[:len(audio)]
    return np.pad(out, (0, len(audio) - len(out)))
