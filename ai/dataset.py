"""AmbuSmart - dataset builder.

Builds 500 ms labelled frames (siren = 1, background = 0) from:
  1. Real recordings in ESC-50 (siren class + 15 urban / environmental classes).
  2. Indian ambulance siren profiles (WAIL, YELP, HI-LO) synthesised with odd
     harmonics, Doppler shift, frequency jitter and synthetic room reverberation,
     mixed onto *real* ESC-50 background frames at SNR -5 ... +20 dB.
  3. Hard negatives: in-band air/car horns, reversing beepers and alarm tones
     that a naive frequency detector would confuse with a siren.

ESC-50 folds keep the split leakage-free: folds 1-3 train, fold 4 validation,
fold 5 test (backgrounds used for mixing follow the same split).
"""
import csv
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, butter, sosfilt, fftconvolve

from features import SR, FRAME_LEN, dual_features, mfcc_mean_vector

ESC = os.environ.get("ESC50", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ESC-50"))
SIREN_TARGET = 42
RNG = np.random.default_rng(42)


# ----------------------------------------------------------------------------- audio io
def load_esc(fname):
    sr, x = wavfile.read(os.path.join(ESC, "audio", fname))
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(1)
    x /= 32768.0
    return resample_poly(x, 160, 441).astype(np.float32)   # 44.1 kHz -> 16 kHz


def frames_of(x, hop=FRAME_LEN // 2, min_rms=3e-3):
    out = []
    for s in range(0, len(x) - FRAME_LEN + 1, hop):
        f = x[s:s + FRAME_LEN]
        if np.sqrt(np.mean(f ** 2)) > min_rms:      # drop silent padding
            out.append(f)
    return out


# ----------------------------------------------------------------------------- sirens
def _square_ish(phase, n_harm=4):
    return sum(np.sin((2 * k + 1) * phase) / (2 * k + 1) for k in range(n_harm))


def synth_siren(kind, dur=FRAME_LEN / SR, rng=RNG):
    """Return a 500 ms window cut from a longer Indian-pattern siren."""
    long = 4.0
    t = np.arange(int(long * SR)) / SR
    doppler = rng.uniform(0.95, 1.05)
    if kind == "wail":                      # 500-800 Hz slow sweep, period 3-5 s
        T = rng.uniform(3, 5)
        f = 500 + 300 * (0.5 - 0.5 * np.cos(2 * np.pi * t / T))
    elif kind == "yelp":                    # 500-1500 Hz, ~4 Hz repetition
        r = rng.uniform(3.2, 4.8)
        saw = (t * r) % 1.0
        f = 500 + 1000 * np.where(saw < 0.5, 2 * saw, 2 - 2 * saw)
    else:                                   # hi-lo 660 / 440 Hz alternation
        r = rng.uniform(0.9, 1.8)
        f = np.where(((t * r) % 1.0) < 0.5, 660.0, 440.0)
    f = f * doppler * (1 + 0.004 * rng.standard_normal(len(t)).cumsum() / np.sqrt(len(t)))
    phase = 2 * np.pi * np.cumsum(f) / SR
    y = _square_ish(phase, n_harm=rng.integers(2, 6))
    # loudspeaker colouration
    lo, hi = rng.uniform(250, 400), rng.uniform(3000, 5000)
    y = sosfilt(butter(2, [lo, hi], btype="band", fs=SR, output="sos"), y)
    # synthetic reverberation (exponentially decaying noise RIR)
    if rng.random() < 0.7:
        rt60 = rng.uniform(0.15, 0.8)
        n = int(rt60 * SR)
        rir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
        rir[0] = 1.0 / rng.uniform(0.3, 1.0)
        y = fftconvolve(y, rir)[: len(t)]
    s = rng.integers(0, len(t) - FRAME_LEN)
    return y[s:s + FRAME_LEN].astype(np.float32)


def synth_hard_negative(kind, rng=RNG):
    t = np.arange(FRAME_LEN) / SR
    if kind == "air_horn":                  # steady harmonic stack 150-500 Hz
        f0 = rng.uniform(150, 500)
        y = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 8))
    elif kind == "car_horn":                # steady two-tone chord
        f1 = rng.uniform(380, 450)
        y = _square_ish(2 * np.pi * f1 * t) + _square_ish(2 * np.pi * f1 * 1.26 * t)
    elif kind == "reverse_beeper":          # 1 kHz on/off beeps
        f0 = rng.uniform(900, 1250)
        gate = (((t * rng.uniform(1.5, 2.5)) % 1.0) < 0.5).astype(float)
        y = np.sin(2 * np.pi * f0 * t) * gate
    else:                                   # alarm: fixed tone pulsed fast
        f0 = rng.uniform(500, 900)
        gate = (((t * rng.uniform(6, 10)) % 1.0) < 0.5).astype(float)
        y = _square_ish(2 * np.pi * f0 * t) * gate
    return y.astype(np.float32)


def mix(sig, bg, snr_db):
    ps = np.mean(sig ** 2) + 1e-12
    pb = np.mean(bg ** 2) + 1e-12
    k = np.sqrt(pb / ps * 10 ** (snr_db / 10))
    return (k * sig + bg).astype(np.float32)


# ----------------------------------------------------------------------------- build split
def build_split(folds, n_synth_pos, n_hard_neg, rng):
    meta = list(csv.DictReader(open(os.path.join(ESC, "meta", "esc50.csv"))))
    avail = set(os.listdir(os.path.join(ESC, "audio")))
    rows = [r for r in meta if r["filename"] in avail and int(r["fold"]) in folds]
    pos, neg, bg_pool = [], [], []
    for r in rows:
        fr = frames_of(load_esc(r["filename"]))
        if int(r["target"]) == SIREN_TARGET:
            pos += [(f, "esc_siren", np.nan) for f in fr]
        else:
            neg += [(f, "esc_" + r["category"], np.nan) for f in fr]
            bg_pool += fr
    bg_pool = np.stack(bg_pool)
    kinds = ["wail", "yelp", "hilo"]
    for i in range(n_synth_pos):
        k = kinds[i % 3]
        snr = rng.uniform(-5, 20)
        bg = bg_pool[rng.integers(len(bg_pool))]
        pos.append((mix(synth_siren(k, rng=rng), bg, snr), "indian_" + k, snr))
    hn = ["air_horn", "car_horn", "reverse_beeper", "alarm"]
    for i in range(n_hard_neg):
        k = hn[i % 4]
        snr = rng.uniform(-5, 20)
        bg = bg_pool[rng.integers(len(bg_pool))]
        neg.append((mix(synth_hard_negative(k, rng=rng), bg, snr), "hard_" + k, snr))
    items = [(a, 1, s, q) for a, s, q in pos] + [(a, 0, s, q) for a, s, q in neg]
    order = rng.permutation(len(items))
    items = [items[i] for i in order]
    LM, MF, Y, SRC, SNR, MV = [], [], [], [], [], []
    for a, y, s, q in items:
        lm, mf = dual_features(a)
        LM.append(lm); MF.append(mf); Y.append(y); SRC.append(s); SNR.append(q)
        MV.append(np.concatenate([mf.mean(1), mf.std(1)]))
    return dict(logmel=np.stack(LM), mfcc=np.stack(MF), y=np.array(Y, np.float32),
                src=np.array(SRC), snr=np.array(SNR, np.float32), mfcc_vec=np.stack(MV))


if __name__ == "__main__":
    if not os.path.isfile(os.path.join(ESC, "meta", "esc50.csv")):
        raise SystemExit(f"ESC-50 not found at {os.path.abspath(ESC)}. "
                         "Download it as described in data/README.md, or set the ESC50 environment variable.")
    os.makedirs("cache", exist_ok=True)
    rng = np.random.default_rng(7)
    for name, folds, npos, nhn in [("train", (1, 2, 3), 3000, 800),
                                   ("val", (4,), 900, 250),
                                   ("test", (5,), 900, 250)]:
        d = build_split(folds, npos, nhn, rng)
        np.savez_compressed(f"cache/{name}.npz", **d)
        print(name, d["logmel"].shape, "pos", int(d["y"].sum()), "neg", int((1 - d["y"]).sum()))
