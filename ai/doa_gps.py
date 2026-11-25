"""AmbuSmart - GCC-PHAT direction-of-arrival and GPS state-estimation experiments.

E4  GCC-PHAT bearing error vs SNR on a 4-mic square array (0.5 m side).
E5  Position error vs time-since-last-packet for three GPS trackers:
      last-fix hold, dead-reckoning (constant velocity) and constant-velocity Kalman filter.
"""
import json
import os
import numpy as np
from features import SR, FRAME_LEN
from dataset import synth_siren

C = 343.0
OUT = "results"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(11)

MICS = 0.25 * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float)   # 0.5 m square
PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]           # 6 pairs


def frac_delay(x, d_samples):
    n = len(x)
    X = np.fft.rfft(x, 2 * n)
    f = np.fft.rfftfreq(2 * n)
    return np.fft.irfft(X * np.exp(-2j * np.pi * f * d_samples), 2 * n)[:n]


def gcc_phat(a, b, interp=16, max_tau=None):
    n = len(a) + len(b)
    A, B = np.fft.rfft(a, n), np.fft.rfft(b, n)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-12
    cc = np.fft.irfft(R, n * interp)
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = min(int(interp * SR * max_tau), max_shift)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    return shift / float(interp * SR)


def estimate_bearing(sig):
    """Least-squares far-field direction from the 6 pairwise TDOAs."""
    rows, taus = [], []
    for i, j in PAIRS:
        tau = gcc_phat(sig[i], sig[j], max_tau=0.75 / C)
        rows.append(MICS[j] - MICS[i]); taus.append(tau * C)
    u, *_ = np.linalg.lstsq(np.array(rows), np.array(taus), rcond=None)
    return np.degrees(np.arctan2(u[1], u[0])) % 360


def simulate_array(theta_deg, snr_db, src=None):
    src = synth_siren(rng.choice(["wail", "yelp", "hilo"]), rng=rng) if src is None else src
    u = np.array([np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))])
    out = []
    p = np.mean(src ** 2)
    for m in MICS:
        delay = -(m @ u) / C * SR                    # plane wave: mic closer to source hears earlier
        s = frac_delay(src, delay)
        # one early wall reflection with random delay/gain (urban facade)
        s = s + rng.uniform(0.1, 0.4) * frac_delay(src, delay + rng.uniform(40, 400))
        noise = rng.standard_normal(len(s)) * np.sqrt(p / 10 ** (snr_db / 10))
        out.append(s + noise)
    return np.array(out)


def angdiff(a, b):
    return (a - b + 180) % 360 - 180


def doa_experiment(snrs=(-10, -5, 0, 5, 10, 15, 20), trials=150):
    res = {}
    for snr in snrs:
        errs = []
        for _ in range(trials):
            th = rng.uniform(0, 360)
            errs.append(angdiff(estimate_bearing(simulate_array(th, snr)), th))
        errs = np.array(errs)
        res[snr] = dict(mae=float(np.mean(np.abs(errs))), median=float(np.median(np.abs(errs))),
                        within5=float(np.mean(np.abs(errs) <= 5)), within30=float(np.mean(np.abs(errs) <= 30)),
                        errors=errs.tolist())
        print(f"DOA SNR {snr:>3} dB: MAE {res[snr]['mae']:.2f} deg, median {res[snr]['median']:.2f}, "
              f"<=5deg {res[snr]['within5']:.2%}, <=30deg {res[snr]['within30']:.2%}")
    return res


# ------------------------------------------------------------------ GPS tracking
def make_track(dur=120, dt=0.25, rng=rng):
    """Ambulance trajectory in local ENU metres with speed changes and turns."""
    n = int(dur / dt)
    pos = np.zeros((n, 2)); v = rng.uniform(8, 16); h = rng.uniform(0, 2 * np.pi)
    for k in range(1, n):
        v = np.clip(v + rng.normal(0, 0.6) * dt, 3, 22)
        if rng.random() < 0.004:                    # occasional 90 deg turn at a junction
            h += rng.choice([-1, 1]) * np.pi / 2
        h += rng.normal(0, 0.01)
        pos[k] = pos[k - 1] + v * dt * np.array([np.cos(h), np.sin(h)])
    return pos


class CVKalman:
    def __init__(self, sigma_gps, q=1.5):
        self.x = None; self.P = None; self.R = np.eye(2) * sigma_gps ** 2; self.q = q

    def predict(self, dt):
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * self.q ** 2

    def update(self, z):
        if self.x is None:
            self.x = np.array([z[0], z[1], 0, 0.]); self.P = np.diag([self.R[0, 0]] * 2 + [100, 100]); return
        H = np.zeros((2, 4)); H[0, 0] = H[1, 1] = 1
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P


def gps_experiment(sigma=10.0, trials=300, gaps=(0, 1, 2, 3, 4, 5, 6, 8, 10), dt=0.25):
    """Error of each tracker `g` seconds after the last received packet."""
    errs = {m: {g: [] for g in gaps} for m in ("hold", "deadreckon", "kalman")}
    for _ in range(trials):
        tr = make_track()
        kf = CVKalman(sigma)
        last_fix, prev_fix, last_t = None, None, None
        warm = 40  # seconds of normal 1 Hz reception before the outage
        for k in range(int((warm - 1) / dt) + 1):
            t = k * dt
            if kf.x is not None:
                kf.predict(dt)
            if abs(t - round(t)) < 1e-9:                      # 1 Hz packet
                z = tr[k] + rng.normal(0, sigma, 2)
                kf.update(z); prev_fix, last_fix, last_t = last_fix, z, t
        k0 = int((warm - 1) / dt)                              # index of last packet
        vel_dr = (last_fix - prev_fix) / 1.0                    # dead-reckoning velocity from last 2 fixes
        xk, Pk = kf.x.copy(), kf.P.copy()
        for g in gaps:
            kf.x, kf.P = xk.copy(), Pk.copy()
            steps = int(g / dt)
            for _ in range(steps):
                kf.predict(dt)
            truth = tr[k0 + steps]
            errs["hold"][g].append(np.linalg.norm(last_fix - truth))
            errs["deadreckon"][g].append(np.linalg.norm(last_fix + vel_dr * g - truth))
            errs["kalman"][g].append(np.linalg.norm(kf.x[:2] - truth))
    res = {m: {g: dict(mean=float(np.mean(v)), p95=float(np.percentile(v, 95)))
               for g, v in d.items()} for m, d in errs.items()}
    for m in res:
        print(m, {g: round(res[m][g]["mean"], 1) for g in gaps})
    return res


if __name__ == "__main__":
    doa = doa_experiment()
    gps_open = gps_experiment(sigma=5.0)
    gps_canyon = gps_experiment(sigma=20.0)
    json.dump(dict(doa=doa, gps_open=gps_open, gps_canyon=gps_canyon),
              open(f"{OUT}/doa_gps.json", "w"), indent=1)
