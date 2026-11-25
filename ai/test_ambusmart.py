"""AmbuSmart unit and integration tests (run: python -m pytest -v)."""
import numpy as np
import pytest

import features as F
import telemetry as T
import doa_gps as G
from dataset import synth_siren, synth_hard_negative, mix

KEY = b"fleet-key-ambusmart-2026"


# ---------------------------------------------------------------- feature extraction
def test_TC01_feature_shapes():
    lm, mf = F.dual_features(np.random.randn(F.FRAME_LEN).astype(np.float32))
    assert lm.shape == (64, 30) and mf.shape == (120, 30)


def test_TC02_level_invariance():
    x = synth_siren("wail")
    a, _ = F.dual_features(x); b, _ = F.dual_features(10 * x)
    assert np.allclose(a, b, atol=1e-3)


def test_TC03_mel_filterbank_covers_siren_band():
    fb = F.mel_filterbank()
    bins = np.fft.rfftfreq(F.N_FFT, 1 / F.SR)
    band = (bins >= 400) & (bins <= 1600)
    assert (fb[:, band].sum(0) > 0).all()


def test_TC04_siren_energy_in_expected_band():
    x = synth_siren("yelp")
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / F.SR)
    assert spec[(f > 450) & (f < 1700)].sum() / spec.sum() > 0.5


def test_TC05_mix_achieves_requested_snr():
    s, b = synth_siren("hilo"), np.random.randn(F.FRAME_LEN).astype(np.float32)
    y = mix(s, b, 10.0)
    sig = y - b
    snr = 10 * np.log10(np.mean(sig ** 2) / np.mean(b ** 2))
    assert abs(snr - 10.0) < 0.1


# ---------------------------------------------------------------- telemetry / security
def test_TC06_packet_roundtrip():
    pkt = T.pack(KEY, 108, 1_764_000_000_000, 13.0306, 77.5649, 92.4, 46.3)
    d = T.verify_and_unpack(KEY, pkt)
    assert d["vehicle_id"] == 108 and abs(d["lat"] - 13.0306) < 1e-9 and d["heading_deg"] == 92.4


def test_TC07_tampered_packet_rejected():
    pkt = bytearray(T.pack(KEY, 108, 1, 13.0, 77.0, 0, 40))
    pkt[14] ^= 0x01                                   # flip one bit of latitude
    assert T.verify_and_unpack(KEY, bytes(pkt)) is None


def test_TC08_wrong_key_rejected():
    pkt = T.pack(b"attacker-key", 108, 1, 13.0, 77.0, 0, 40)
    assert T.verify_and_unpack(KEY, pkt) is None


def test_TC09_truncated_packet_rejected():
    assert T.verify_and_unpack(KEY, T.pack(KEY, 1, 1, 0, 0, 0, 0)[:-1]) is None


# ---------------------------------------------------------------- bearing estimation
def test_TC10_gcc_phat_recovers_known_delay():
    x = np.random.randn(8000)
    y = G.frac_delay(x, 7.0)
    tau = G.gcc_phat(y, x)
    assert abs(tau * F.SR - 7.0) < 0.2


@pytest.mark.parametrize("theta", [10, 100, 200, 300])
def test_TC11_bearing_high_snr(theta):
    G.rng = np.random.default_rng(theta)
    est = G.estimate_bearing(G.simulate_array(theta, 20))
    assert abs(G.angdiff(est, theta)) <= 5


# ---------------------------------------------------------------- Kalman tracker
def test_TC12_kalman_converges_on_straight_track():
    kf = G.CVKalman(sigma_gps=5.0)
    rng = np.random.default_rng(0)
    for t in range(30):
        if kf.x is not None:
            kf.predict(1.0)
        kf.update(np.array([12.0 * t, 0.0]) + rng.normal(0, 5, 2))
    # speed within 20 % and position within 10 m of truth after 30 noisy fixes
    assert abs(kf.x[2] - 12.0) < 2.4 and abs(kf.x[3]) < 2.4 and abs(kf.x[0] - 12.0 * 29) < 10


def test_TC13_kalman_extrapolation_bounded():
    kf = G.CVKalman(sigma_gps=5.0)
    for t in range(20):
        if kf.x is not None:
            kf.predict(1.0)
        kf.update(np.array([11.1 * t, 0.0]))
    for _ in range(12):
        kf.predict(0.25)                               # 3 s outage
    truth = 11.1 * 22
    assert abs(kf.x[0] - truth) < 33                  # paper's 33 m bound at 40 km/h


# ---------------------------------------------------------------- fusion logic (canonical cases)
fs = pytest.importorskip("fusion_sim")


def _ep(score, gps_ok, gps_dist, closing, doa, gps_bear, arms=(0, 90, 180, 270)):
    n = len(score)
    return dict(score=np.array(score, float), gps_ok=np.array(gps_ok, bool), gps_dist=np.array(gps_dist, float),
                closing=np.array(closing, bool), doa=np.array(doa, float), gps_bear=np.array(gps_bear, float),
                n=n, arms=list(arms), dist=np.array(gps_dist, float), v=10)


def test_TC14_gps_path_fires_with_sanity_check():
    e = _ep([0.1, 0.65, 0.7, 0.2, 0.2], [1] * 5, [400, 350, 320, 290, 280], [1] * 5, [90] * 5, [90] * 5)
    assert fs.p4(e) == 3                               # siren seen at 350-320 m, fires on entering 300 m
    e_wrong_dir = _ep([0.1, 0.65, 0.7, 0.2, 0.2], [1] * 5, [400, 350, 320, 290, 280], [1] * 5, [250] * 5, [90] * 5)
    assert fs.p4(e_wrong_dir) is None                  # siren heard from another direction


def test_TC15_gps_path_blocked_without_siren():
    e = _ep([0.1] * 8, [1] * 8, [250] * 8, [1] * 8, [90] * 8, [90] * 8)
    assert fs.p4(e) is None


def test_TC16_acoustic_path_needs_three_frames_and_bearing():
    e = _ep([0.9, 0.9, 0.9, 0.9], [1] * 4, [500] * 4, [1] * 4, [92, 95, 88, 90], [90] * 4)
    assert fs.p4(e) == 2
    e_bad = _ep([0.9] * 4, [1] * 4, [500] * 4, [1] * 4, [200] * 4, [90] * 4)
    assert fs.p4(e_bad) is None


def test_TC17_fallback_needs_five_frames_at_090():
    n = 6
    e = _ep([0.95] * n, [0] * n, [np.inf] * n, [0] * n, [181] * n, [np.nan] * n)
    assert fs.p4(e) == 4
    e2 = _ep([0.88] * n, [0] * n, [np.inf] * n, [0] * n, [181] * n, [np.nan] * n)
    assert fs.p4(e2) is None


def test_TC18_horn_burst_does_not_trigger_fallback():
    e = _ep([0.97, 0.97, 0.2, 0.97, 0.97, 0.1], [0] * 6, [np.inf] * 6, [0] * 6, [0] * 6, [np.nan] * 6)
    assert fs.p4(e) is None


# ---------------------------------------------------------------- trained model
def test_TC19_trained_model_separates_siren_and_horn():
    import pickle
    model = pickle.load(open("results/ambusmart_siren_model.pkl", "rb"))
    rng = np.random.default_rng(5)
    bg = np.random.default_rng(1).standard_normal(F.FRAME_LEN).astype(np.float32) * 0.05
    score = lambda x: model.predict_proba(F.mfcc_mean_vector(x)[None])[0, 1]
    sir = np.mean([score(mix(synth_siren(k, rng=rng), bg, 15)) for k in ("wail", "yelp", "hilo")])
    horn = np.mean([score(mix(synth_hard_negative(k, rng=rng), bg, 15)) for k in ("air_horn", "car_horn")])
    assert sir > 0.8 and horn < 0.5
