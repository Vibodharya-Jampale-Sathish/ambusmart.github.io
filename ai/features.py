"""AmbuSmart - acoustic feature extraction.

Dual-path features for each 500 ms frame (16 kHz):
  * log-Mel spectrogram  : 512-pt STFT, hop 256, Hann window, 64 Mel bands
  * MFCC + delta + delta2: 40 cepstral coefficients each -> 120 rows
The two maps are stacked along the frequency axis -> (184, T) single-channel image.
Implemented with NumPy/SciPy only so it can be ported to an embedded target.
"""
import numpy as np
from scipy.fft import dct

SR = 16000
FRAME_SEC = 0.5
FRAME_LEN = int(SR * FRAME_SEC)          # 8000 samples
N_FFT = 512
HOP = 256
N_MELS = 64
N_MFCC = 40


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=50.0, fmax=None):
    fmax = fmax or sr / 2
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz = _mel_to_hz(mels)
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(bins)), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, c, hi = hz[m - 1], hz[m], hz[m + 1]
        up = (bins - lo) / (c - lo)
        down = (hi - bins) / (hi - c)
        fb[m - 1] = np.maximum(0, np.minimum(up, down))
    # Slaney-style area normalisation
    fb *= (2.0 / (hz[2:] - hz[:-2]))[:, None]
    return fb


_FB = mel_filterbank()
_WIN = np.hanning(N_FFT).astype(np.float32)


def stft_mag(x):
    n_frames = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = x[idx] * _WIN
    return np.abs(np.fft.rfft(frames, axis=1)).T          # (257, T)


def delta(feat, width=2):
    T = feat.shape[1]
    pad = np.pad(feat, ((0, 0), (width, width)), mode="edge")
    num = sum(n * (pad[:, width + n:width + n + T] - pad[:, width - n:width - n + T])
              for n in range(1, width + 1))
    return num / (2 * sum(n * n for n in range(1, width + 1)))


def logmel(x):
    S = stft_mag(x) ** 2
    M = _FB @ S
    return np.log(M + 1e-6).astype(np.float32)            # (64, T)


def mfcc_stack(lm):
    c = dct(lm, type=2, axis=0, norm="ortho")[:N_MFCC]
    return np.vstack([c, delta(c), delta(delta(c))]).astype(np.float32)   # (120, T)


def dual_features(x):
    """Return the stacked (184, T) dual-path feature map for one 500 ms frame."""
    x = x.astype(np.float32)
    x = x - x.mean()
    peak = np.abs(x).max() + 1e-9
    x = x / peak                                           # level-invariant input
    lm = logmel(x)
    mf = mfcc_stack(lm)
    return lm, mf


def mfcc_mean_vector(x):
    """120-d utterance-level MFCC statistics for the classical baseline."""
    lm, mf = dual_features(x)
    return np.concatenate([mf.mean(1), mf.std(1)])
