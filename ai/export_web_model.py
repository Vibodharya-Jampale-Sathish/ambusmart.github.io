"""AmbuSmart - export the trained siren classifier for the browser dashboard.

Writes dashboard/src/siren/siren_model.json with the StandardScaler statistics and the
MLP weights, so the dashboard can run the *same* model live on microphone audio.
Run after train.py:  python export_web_model.py
"""
import json
import os
import pickle

import numpy as np

from features import SR, FRAME_LEN, N_FFT, HOP, N_MELS, N_MFCC

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "results", "ambusmart_siren_model.pkl")
DST = os.path.join(HERE, "..", "dashboard", "src", "siren", "siren_model.json")


def r(a):
    """Round to float32 precision to keep the JSON small."""
    return np.round(np.asarray(a, dtype=np.float64), 7).tolist()


def main():
    pipe = pickle.load(open(SRC, "rb"))
    scaler, mlp = pipe.steps[0][1], pipe.steps[-1][1]
    if type(mlp).__name__ != "MLPClassifier":
        raise SystemExit(f"Only an MLPClassifier can be exported, the saved model is {type(mlp).__name__}.")
    assert mlp.activation == "relu" and mlp.out_activation_ == "logistic"
    model = dict(
        name="AmbuSmart MLP (128-64)",
        features=dict(sr=SR, frame_len=FRAME_LEN, n_fft=N_FFT, hop=HOP, n_mels=N_MELS, n_mfcc=N_MFCC),
        scaler=dict(mean=r(scaler.mean_), scale=r(scaler.scale_)),
        # weights[l][i][j]: input i -> unit j, as in sklearn coefs_
        weights=[r(w) for w in mlp.coefs_],
        biases=[r(b) for b in mlp.intercepts_],
    )
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with open(DST, "w") as f:
        json.dump(model, f, separators=(",", ":"))
    print(f"wrote {os.path.relpath(DST, HERE)} ({os.path.getsize(DST) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
