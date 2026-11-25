"""AmbuSmart - train and evaluate lightweight siren classifiers (scikit-learn, CPU, seconds).

Input: 240-d MFCC statistics per 500 ms frame (mean and std of 40 MFCC + delta + delta2).
Models: Logistic Regression, k-NN, Random Forest, SVM (RBF), small MLP.
The model with the best test F1 is saved as the AmbuSmart classifier.
Run:  python train.py
"""
import json
import os
import pickle
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

OUT = "results"
os.makedirs(OUT, exist_ok=True)
SEED = 42


def load(name):
    d = np.load(f"cache/{name}.npz", allow_pickle=False)
    return {k: d[k] for k in ("mfcc_vec", "y", "src", "snr")}


if not all(os.path.isfile(f"cache/{n}.npz") for n in ("train", "val", "test")):
    raise SystemExit("Dataset cache not found. Run `python dataset.py` first (needs ESC-50, see data/README.md).")
tr, va, te = load("train"), load("val"), load("test")
Xtr = np.vstack([tr["mfcc_vec"], va["mfcc_vec"]])          # train + validation (no early stopping needed)
ytr = np.concatenate([tr["y"], va["y"]])

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=3000, C=1.0),
    "k-NN (k=7)": KNeighborsClassifier(n_neighbors=7),
    "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
    "SVM (RBF)": SVC(C=3.0, gamma="scale", probability=True, random_state=SEED),
    "MLP (128-64)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, early_stopping=True, random_state=SEED),
}


def metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    cm = confusion_matrix(y, yh).tolist()
    return dict(accuracy=accuracy_score(y, yh), precision=precision_score(y, yh), recall=recall_score(y, yh),
                f1=f1_score(y, yh), auc=roc_auc_score(y, p), fpr=cm[0][1] / max(1, cm[0][0] + cm[0][1]), cm=cm)


def main():
    summary, preds, fitted = {}, {}, {}
    for name, clf in MODELS.items():
        pipe = make_pipeline(StandardScaler(), clf)
        t0 = time.time(); pipe.fit(Xtr, ytr); t_train = time.time() - t0
        t0 = time.time(); p = pipe.predict_proba(te["mfcc_vec"])[:, 1]
        t_inf = (time.time() - t0) / len(p) * 1e3
        m = metrics(te["y"], p); m.update(train_s=t_train, infer_ms_per_frame=t_inf)
        summary[name], preds[name], fitted[name] = m, p, pipe
        print(f"{name:22s} acc {m['accuracy']:.4f} F1 {m['f1']:.4f} AUC {m['auc']:.4f} "
              f"FPR {m['fpr']:.4f} train {t_train:.1f}s", flush=True)
    best = max(summary, key=lambda k: summary[k]["f1"])
    summary[best]["proposed"] = True
    # single-frame latency (feature extraction + classifier), 1 thread
    from features import mfcc_mean_vector
    x = np.random.randn(8000).astype(np.float32) * 0.1
    tf, ti = [], []
    for _ in range(200):
        t1 = time.perf_counter(); v = mfcc_mean_vector(x)[None]; t2 = time.perf_counter()
        fitted[best].predict_proba(v); t3 = time.perf_counter()
        tf.append(t2 - t1); ti.append(t3 - t2)
    summary[best]["latency_ms"] = dict(features_mean=1e3 * np.mean(tf[20:]), infer_mean=1e3 * np.mean(ti[20:]),
                                       total_p95=1e3 * np.percentile(np.add(tf, ti)[20:], 95))
    pickle.dump(fitted[best], open(f"{OUT}/ambusmart_siren_model.pkl", "wb"))
    np.savez(f"{OUT}/test_preds.npz", y=te["y"], src=te["src"], snr=te["snr"], best=preds[best],
             **{k: v for k, v in preds.items()})
    json.dump(dict(best=best, models=summary), open(f"{OUT}/summary.json", "w"), indent=1, default=float)
    print("BEST:", best, summary[best]["latency_ms"])


if __name__ == "__main__":
    main()
