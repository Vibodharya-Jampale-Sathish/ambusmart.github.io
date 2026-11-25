"""AmbuSmart - Monte-Carlo evaluation of the activation (fusion) policies.

Per-frame acoustic confidences are drawn from the *trained* Dual-path CNN's
scores on the held-out test fold (binned by SNR), with temporal correlation
added through an AR(1) Gaussian copula. Bearing errors are drawn from the
GCC-PHAT experiment; GPS positions come from the Kalman tracker model.

Scenarios (each episode simulated on a 250 ms update grid)
  S1  genuine approach, GPS nominal            (packet loss 5 %, sigma 5 m)
  S2  genuine approach, urban canyon            (packet loss 30 % in bursts, sigma 20 m)
  S3  genuine approach, cellular outage         (no GPS at all)
  S4  ambulance on an adjacent street           (siren audible, not approaching this junction)
  S5  no ambulance - 60 s of urban noise with in-band horn/alarm events
  S6  stale GPS broadcast, siren OFF            (vehicle in zone, not on an emergency run)

Policies
  P1  acoustic-only, single frame c >= 0.5
  P2  acoustic-only, c >= 0.85 for 3 frames (persistence)
  P3  GPS-only, inside 300 m and approaching
  P4  AmbuSmart fusion (GPS-primary / acoustic-primary / acoustic-only fallback)
"""
import json
import os
import numpy as np
from scipy.stats import norm

OUT = "results"
rng = np.random.default_rng(2026)
DT = 0.25

# ------------------------------------------------------------------ score pools from the trained CNN
SNR_BINS = np.arange(-5, 21, 5)                      # -5,0,5,10,15,20
POS_POOL = NEG_POOL = HARD_POOL = DOA = DOA_SNRS = None


def load_pools():
    global POS_POOL, NEG_POOL, HARD_POOL, DOA, DOA_SNRS
    P = np.load(f"{OUT}/test_preds.npz", allow_pickle=False)
    scores = P["best"]
    y, src, snr = P["y"], P["src"], P["snr"]
    indian = np.char.startswith(src.astype(str), "indian_")
    POS_POOL = {b: np.sort(scores[indian & (np.abs(snr - b) <= 2.5)]) for b in SNR_BINS}
    NEG_POOL = np.sort(scores[y == 0])
    HARD_POOL = np.sort(scores[np.char.startswith(src.astype(str), "hard_")])
    DOA = json.load(open(f"{OUT}/doa_gps.json"))["doa"]
    DOA_SNRS = np.array(sorted(int(k) for k in DOA))


def pool_for(snr_db):
    if snr_db < -7.5:
        return NEG_POOL                                # siren inaudible
    b = SNR_BINS[np.argmin(np.abs(SNR_BINS - snr_db))]
    return POS_POOL[b]


def ar_uniforms(n, rho=0.7):
    z = np.empty(n); z[0] = rng.standard_normal()
    for k in range(1, n):
        z[k] = rho * z[k - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal()
    return norm.cdf(z)


def sample(pool, u):
    return pool[min(int(u * len(pool)), len(pool) - 1)]


def doa_error(snr_db):
    s = DOA_SNRS[np.argmin(np.abs(DOA_SNRS - snr_db))]
    e = DOA[str(s)]["errors"]
    return e[rng.integers(len(e))]


def angdiff(a, b):
    return (a - b + 180) % 360 - 180


# ------------------------------------------------------------------ episode generator
def episode(kind):
    """Return per-frame arrays: score, doa_bearing, gps_ok, gps_dist, gps_closing, gps_bearing,
    true_dist (np.inf if no ambulance approaching), speed."""
    if kind in ("S1", "S2", "S3", "S6"):
        v = rng.uniform(25, 60) / 3.6
        d0 = 700.0
        n = int((d0 + 30) / v / DT)
        dist = d0 - v * DT * np.arange(n)
        approach_bearing = rng.choice([0, 90, 180, 270]) + rng.normal(0, 3)
        true_bearing = np.full(n, approach_bearing)
    elif kind == "S4":
        # parallel street 60-150 m away; ambulance drives past, never toward the junction
        v = rng.uniform(25, 60) / 3.6
        off = rng.uniform(60, 150)
        n = int(80 / DT)
        along = -400 + v * DT * np.arange(n)
        dist = np.hypot(along, off)
        approach_bearing = None
        true_bearing = np.degrees(np.arctan2(off, along)) % 360
    else:  # S5
        n = int(60 / DT)
        dist = np.full(n, np.inf); v = 0; true_bearing = rng.uniform(0, 360, n)
        approach_bearing = None

    # ---- acoustic channel
    ref_snr = rng.uniform(8, 22)                        # SNR at 25 m, depends on traffic noise
    u = ar_uniforms(n)
    sc = np.empty(n); doa = np.empty(n); snr_t = np.empty(n)
    siren_on = kind != "S6"
    hard_events = np.zeros(n, bool)
    if kind == "S5":
        for _ in range(rng.poisson(6)):                 # ~6 horn / alarm bursts per minute
            s = rng.integers(n); hard_events[s:s + rng.integers(2, 12)] = True
    for k in range(n):
        if kind == "S5":
            pool = HARD_POOL if hard_events[k] else NEG_POOL
            sc[k] = sample(pool, u[k]); doa[k] = rng.uniform(0, 360); snr_t[k] = -99
        elif not siren_on:
            sc[k] = sample(NEG_POOL, u[k]); doa[k] = rng.uniform(0, 360); snr_t[k] = -99
        else:
            s_db = ref_snr - 20 * np.log10(max(dist[k], 5) / 25)
            snr_t[k] = s_db
            sc[k] = sample(pool_for(s_db), u[k])
            doa[k] = (true_bearing[k] + (doa_error(s_db) if s_db > -12 else rng.uniform(-180, 180))) % 360

    # ---- GPS channel
    gps_ok = np.zeros(n, bool); gps_dist = np.full(n, np.inf)
    gps_closing = np.zeros(n, bool); gps_bear = np.full(n, np.nan)
    if kind in ("S1", "S2", "S4", "S6"):
        sigma, loss = (20.0, 0.30) if kind == "S2" else (5.0, 0.05)
        last_pkt_t, est, last_d = -99.0, None, None
        burst = 0
        for k in range(n):
            t = k * DT
            if abs(t - round(t)) < 1e-9:
                if burst > 0:
                    burst -= 1
                elif rng.random() < loss:
                    burst = rng.integers(1, 6) if kind == "S2" else 0
                else:
                    last_pkt_t = t
            age = t - last_pkt_t
            if age <= 10:
                gps_ok[k] = True
                # Kalman-extrapolated distance: truth + error growing with packet age
                err = rng.normal(0, sigma * 0.6) + rng.normal(0, 1.2 * age)
                gps_dist[k] = max(0.0, dist[k] + err)
                if kind == "S4":
                    gps_closing[k] = False                      # heading not toward junction
                    gps_bear[k] = true_bearing[k]
                else:
                    gps_closing[k] = True
                    gps_bear[k] = approach_bearing
    return dict(score=sc, doa=doa, gps_ok=gps_ok, gps_dist=gps_dist, closing=gps_closing,
                gps_bear=gps_bear, dist=dist, v=v, n=n,
                arms=[0, 90, 180, 270] if approach_bearing is None else
                [approach_bearing % 360, (approach_bearing + 90) % 360,
                 (approach_bearing + 180) % 360, (approach_bearing + 270) % 360])


# ------------------------------------------------------------------ policies -> first activation frame
def p1(e):
    idx = np.flatnonzero(e["score"] >= 0.5)
    return idx[0] if len(idx) else None


def run_len_hit(mask, n_req):
    c = 0
    for k, m in enumerate(mask):
        c = c + 1 if m else 0
        if c >= n_req:
            return k
    return None


def p2(e):
    return run_len_hit(e["score"] >= 0.85, 3)


def p3(e):
    m = e["gps_ok"] & (e["gps_dist"] <= 300) & e["closing"]
    idx = np.flatnonzero(m)
    return idx[0] if len(idx) else None


def p4(e):
    sc, n = e["score"], e["n"]
    last_ok = -999
    run2 = run3 = run5 = 0
    for k in range(n):
        bearing_ok_gps = (not np.isnan(e["gps_bear"][k]) and
                          abs(angdiff(e["doa"][k], e["gps_bear"][k])) <= 30)
        bearing_ok_arm = min(abs(angdiff(e["doa"][k], a)) for a in e["arms"]) <= 30
        # GPS-primary sanity check: c_a >= 0.60 with bearing matching the GPS approach on 2 consecutive frames
        run2 = run2 + 1 if (sc[k] >= 0.60 and bearing_ok_gps) else 0
        if run2 >= 2:
            last_ok = k
        # GPS-primary path: inside 300 m, closing, sanity check satisfied within the last 5 s
        if (e["gps_ok"][k] and e["gps_dist"][k] <= 300 and e["closing"][k]
                and (k - last_ok) * DT <= 5.0):
            return k
        # acoustic-primary path (GPS present, outside zone): high confidence + bearing agrees with GPS
        run3 = run3 + 1 if (sc[k] >= 0.85 and bearing_ok_gps and e["closing"][k]) else 0
        if e["gps_ok"][k] and run3 >= 3:
            return k
        # acoustic-only fallback (GPS unavailable > 10 s): stricter threshold, bearing on an approach arm
        run5 = run5 + 1 if (sc[k] >= 0.90 and bearing_ok_arm) else 0
        if not e["gps_ok"][k] and run5 >= 5:
            return k
    return None


POLICIES = {"P1 Acoustic-only (single frame)": p1, "P2 Acoustic-only (3-frame persistence)": p2,
            "P3 GPS-only": p3, "P4 AmbuSmart fusion (proposed)": p4}


def evaluate(n_ep=1500):
    load_pools()
    res = {}
    for kind in ["S1", "S2", "S3", "S4", "S5", "S6"]:
        eps = [episode(kind) for _ in range(n_ep)]
        res[kind] = {}
        for name, f in POLICIES.items():
            hits, lead_t, lead_d = 0, [], []
            for e in eps:
                k = f(e)
                if kind in ("S1", "S2", "S3"):
                    # a useful activation must arrive while the ambulance is still >= 30 m out
                    if k is not None and e["dist"][k] >= 30:
                        hits += 1; lead_d.append(e["dist"][k]); lead_t.append(e["dist"][k] / e["v"])
                else:
                    if k is not None:
                        hits += 1
            r = dict(rate=hits / n_ep)
            if lead_t:
                r.update(lead_time_s=float(np.median(lead_t)), lead_dist_m=float(np.median(lead_d)),
                         lead_time_p10=float(np.percentile(lead_t, 10)))
            res[kind][name] = r
            print(kind, name, r)
    # combined headline numbers
    head = {}
    for name in POLICIES:
        det = np.mean([res[s][name]["rate"] for s in ("S1", "S2", "S3")])
        fa = np.mean([res[s][name]["rate"] for s in ("S4", "S5", "S6")])
        n_pos, n_neg = 3 * n_ep, 3 * n_ep
        correct = det * n_pos + (1 - fa) * n_neg
        head[name] = dict(detection_rate=float(det), false_activation_rate=float(fa),
                          decision_accuracy=float(correct / (n_pos + n_neg)),
                          false_per_hour_S5=float(res["S5"][name]["rate"] * 60))
    res["headline"] = head
    return res


if __name__ == "__main__":
    r = evaluate()
    json.dump(r, open(f"{OUT}/fusion_sim.json", "w"), indent=1)
    for k, v in r["headline"].items():
        print(f"{k:40s} det {v['detection_rate']:.3f} FA {v['false_activation_rate']:.3f} "
              f"acc {v['decision_accuracy']:.3f} FA/h(noise) {v['false_per_hour_S5']:.1f}")
