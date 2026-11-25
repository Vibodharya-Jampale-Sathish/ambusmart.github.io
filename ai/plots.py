"""Result figures for the AmbuSmart report and paper."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

R, F = "results", "figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6"})
S_ = json.load(open(f"{R}/summary.json"))
BEST = S_["best"]
_pal = ["#9aa0a6", "#5f6368", "#2e6da4", "#d68a1c", "#5b8c5a"]
COL = {k: c for k, c in zip([m for m in S_["models"] if m != BEST], _pal)}
COL[BEST] = "#c0392b"


def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{F}/{name}.png", dpi=200); plt.close(fig)


def training_curves():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for mode, lab in [("mel", "CNN (log-Mel only)"), ("mfcc", "CNN (MFCC only)"), ("dual", "CNN Dual-path (proposed)")]:
        h = json.load(open(f"{R}/history_{mode}.json"))
        e = [x["epoch"] for x in h]
        ax[0].plot(e, [x["val_loss"] for x in h], color=COL[lab], label=lab)
        ax[1].plot(e, [x["val_acc"] for x in h], color=COL[lab], label=lab)
        if mode == "dual":
            ax[0].plot(e, [x["train_loss"] for x in h], color=COL[lab], ls="--", label="Dual-path (train)")
            ax[1].plot(e, [x["train_acc"] for x in h], color=COL[lab], ls="--", label="Dual-path (train)")
    ax[0].set_title("Loss per epoch", loc="left"); ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Binary cross-entropy")
    ax[1].set_title("Accuracy per epoch", loc="left"); ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy")
    ax[1].legend(fontsize=8, frameon=False, loc="lower right")
    save(fig, "res_training_curves")


def confusion(summary):
    cm = np.array(summary["models"][BEST]["cm"])
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(cm, cmap="Blues"); ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "#1f2a44")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Background", "Siren"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Background", "Siren"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{BEST}, test fold", loc="left")
    save(fig, "res_confusion")


def roc(P):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for k, c in COL.items():
        fpr, tpr, _ = roc_curve(P["y"], P[k])
        ax.plot(fpr, tpr, color=c, lw=2.2 if k == BEST else 1.3, label=k + (" (proposed)" if k == BEST else ""))
    ax.plot([0, 1], [0, 1], color="#bbb", ls=":")
    ax.set_xlim(0, 0.3); ax.set_ylim(0.6, 1.0)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (zoomed)", loc="left"); ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    save(fig, "res_roc")


def snr_curve(P):
    y, src, snr = P["y"], P["src"].astype(str), P["snr"]
    m = np.char.startswith(src, "indian_")
    bins = [(-5, 0), (0, 5), (5, 10), (10, 15), (15, 20)]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for k in COL:
        p = P[k]
        rec = [np.mean(p[m & (snr >= a) & (snr < b)] >= 0.5) for a, b in bins]
        ax.plot([f"{a} to {b}" for a, b in bins], rec, marker="o", color=COL[k], label=k,
                lw=2.2 if k == BEST else 1.3)
    ax.set_xlabel("Siren-to-background SNR (dB)"); ax.set_ylabel("Detection rate (recall)")
    pass
    ax.legend(fontsize=8, frameon=False, loc="lower right"); ax.set_ylim(0.0, 1.02)
    save(fig, "res_snr")


def per_source(P):
    src, y = P["src"].astype(str), P["y"]
    p = P["best"]
    groups = {"Real siren (ESC-50)": src == "esc_siren", "Indian WAIL": src == "indian_wail",
              "Indian YELP": src == "indian_yelp", "Indian HI-LO": src == "indian_hilo",
              "Air horn": src == "hard_air_horn", "Car horn": src == "hard_car_horn",
              "Reverse beeper": src == "hard_reverse_beeper", "Alarm tone": src == "hard_alarm",
              "Real urban/env. noise": np.char.startswith(src, "esc_") & (src != "esc_siren")}
    names, acc, cols = [], [], []
    for n, m in groups.items():
        names.append(n); acc.append(np.mean((p[m] >= 0.5) == y[m]))
        cols.append("#c0392b" if y[m].mean() > 0.5 else "#2e6da4")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(names, acc, color=cols); ax.invert_yaxis(); ax.set_xlim(0.5, 1.0)
    for i, a in enumerate(acc):
        ax.text(a + 0.004, i, f"{a:.1%}", va="center", fontsize=8.5)
    ax.set_xlabel("Correct decisions"); ax.set_title(f"{BEST} accuracy by sound source (red = siren, blue = background)", loc="left", fontsize=10)
    save(fig, "res_per_source")
    return dict(zip(names, map(float, acc)))


def doa_gps(D):
    doa = D["doa"]
    snrs = sorted(int(k) for k in doa)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(snrs, [doa[str(s)]["median"] for s in snrs], marker="o", color="#c0392b", label="Median |error|")
    ax[0].plot(snrs, [doa[str(s)]["mae"] for s in snrs], marker="s", color="#2e6da4", label="Mean |error|")
    ax[0].axhline(5, color="#999", ls=":"); ax[0].text(snrs[0], 6, "5 deg target", fontsize=8, color="#666")
    ax[0].set_xlabel("SNR at each microphone (dB)"); ax[0].set_ylabel("Bearing error (degrees)")
    ax[0].set_title("GCC-PHAT direction-of-arrival", loc="left"); ax[0].legend(frameon=False, fontsize=8)
    for key, ls in [("gps_open", "-"), ("gps_canyon", "--")]:
        g = D[key]
        gaps = sorted(float(k) for k in g["kalman"])
        for m, c, lab in [("hold", "#9aa0a6", "Last-fix hold"), ("deadreckon", "#d68a1c", "Dead-reckoning"),
                          ("kalman", "#c0392b", "Kalman (CV)")]:
            ax[1].plot(gaps, [g[m][str(int(x))]["mean"] for x in gaps], ls=ls, color=c,
                       label=f"{lab}, {'open sky' if key == 'gps_open' else 'urban canyon'}")
    ax[1].set_xlabel("Seconds since last GPS packet"); ax[1].set_ylabel("Mean position error (m)")
    ax[1].set_title("GPS extrapolation error", loc="left"); ax[1].legend(fontsize=7, frameon=False)
    save(fig, "res_doa_gps")


def fusion(FS):
    pols = list(FS["S1"].keys())
    short = ["P1 Acoustic\nsingle frame", "P2 Acoustic\n3-frame", "P3 GPS-only", "P4 AmbuSmart\nfusion"]
    scen = [("S1", "Approach,\nGPS nominal"), ("S2", "Approach,\nurban canyon"), ("S3", "Approach,\nno GPS")]
    neg = [("S4", "Adjacent-street\nambulance"), ("S5", "Urban noise\n(60 s)"), ("S6", "GPS on,\nsiren off")]
    cols = ["#9aa0a6", "#5f6368", "#2e6da4", "#c0392b"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    w = 0.2
    for i, p in enumerate(pols):
        ax[0].bar(np.arange(3) + (i - 1.5) * w, [FS[s][p]["rate"] for s, _ in scen], w, color=cols[i], label=short[i].replace("\n", " "))
        ax[1].bar(np.arange(3) + (i - 1.5) * w, [FS[s][p]["rate"] for s, _ in neg], w, color=cols[i])
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels([l for _, l in scen], fontsize=8.5)
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels([l for _, l in neg], fontsize=8.5)
    ax[0].set_ylabel("Timely activation rate"); ax[1].set_ylabel("False activation rate")
    ax[0].set_title("Genuine ambulance approaches (higher is better)", loc="left", fontsize=10)
    ax[1].set_title("No approaching ambulance (lower is better)", loc="left", fontsize=10)
    ax[0].legend(fontsize=8, frameon=False, loc="lower left"); ax[0].set_ylim(0, 1.05); ax[1].set_ylim(0, 1.05)
    save(fig, "res_fusion")


def feature_example():
    from dataset import synth_siren, load_esc, frames_of
    from features import dual_features
    rng = np.random.default_rng(3)
    s = synth_siren("yelp", rng=rng)
    import os
    from dataset import ESC
    if not os.path.isdir(os.path.join(ESC, "audio")):
        print("Skipping res_features.png: ESC-50 not found (see data/README.md)")
        return
    eng = sorted(f for f in os.listdir(os.path.join(ESC, "audio")) if f.endswith("-44.wav"))[0]
    bg = frames_of(load_esc(eng))[3]
    from dataset import mix
    x = mix(s, bg, 5)
    lm, mf = dual_features(x)
    lb, mb = dual_features(bg)
    fig, ax = plt.subplots(2, 2, figsize=(10, 5.2))
    for a, im, t in [(ax[0, 0], lm, "log-Mel, siren + traffic (5 dB)"), (ax[0, 1], lb, "log-Mel, background only"),
                     (ax[1, 0], mf[:40], "MFCC (40), siren + traffic"), (ax[1, 1], mb[:40], "MFCC (40), background only")]:
        a.imshow(im, origin="lower", aspect="auto", cmap="magma"); a.grid(False)
        a.set_title(t, loc="left", fontsize=9.5); a.set_xlabel("Frame (16 ms hop)")
    ax[0, 0].set_ylabel("Mel band"); ax[1, 0].set_ylabel("Coefficient")
    save(fig, "res_features")


if __name__ == "__main__":
    S = json.load(open(f"{R}/summary.json"))
    P = dict(np.load(f"{R}/test_preds.npz", allow_pickle=False))
    P = {k: v for k, v in P.items()}
    confusion(S); roc(P); snr_curve(P)
    ps = per_source(P)
    json.dump(ps, open(f"{R}/per_source.json", "w"), indent=1)
    doa_gps(json.load(open(f"{R}/doa_gps.json")))
    fusion(json.load(open(f"{R}/fusion_sim.json")))
    feature_example()
    print(ps)
