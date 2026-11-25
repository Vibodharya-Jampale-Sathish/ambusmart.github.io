// AmbuSmart - browser port of ai/features.py + the trained MLP siren classifier.
// Feature pipeline is identical to Python: 500 ms frame @ 16 kHz -> 512-pt STFT (hop 256, Hann)
// -> 64 Mel bands -> log -> 40 MFCC + delta + delta2 -> mean & std over time (240-d)
// -> StandardScaler -> MLP 240-128-64-1 (ReLU, sigmoid output) -> siren confidence c_a.
import MODEL from './siren_model.json';

const { sr: SR, frame_len: FRAME_LEN, n_fft: N_FFT, hop: HOP, n_mels: N_MELS, n_mfcc: N_MFCC } = MODEL.features;
export { SR, FRAME_LEN };

const N_BINS = N_FFT / 2 + 1;
const N_FRAMES = 1 + Math.floor((FRAME_LEN - N_FFT) / HOP); // 30

// ---- constants (same formulas as features.py)
const hzToMel = (f) => 2595 * Math.log10(1 + f / 700);
const melToHz = (m) => 700 * (10 ** (m / 2595) - 1);

const WIN = new Float64Array(N_FFT).map((_, n) => 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (N_FFT - 1))); // np.hanning

const MEL_FB = (() => {
  const lo = hzToMel(50), hi = hzToMel(SR / 2);
  const hz = Array.from({ length: N_MELS + 2 }, (_, i) => melToHz(lo + ((hi - lo) * i) / (N_MELS + 1)));
  const fb = Array.from({ length: N_MELS }, () => new Float64Array(N_BINS));
  for (let m = 1; m <= N_MELS; m++) {
    const [a, c, b] = [hz[m - 1], hz[m], hz[m + 1]];
    const norm = 2 / (b - a); // Slaney-style area normalisation
    for (let k = 0; k < N_BINS; k++) {
      const f = (k * SR) / N_FFT;
      fb[m - 1][k] = Math.max(0, Math.min((f - a) / (c - a), (b - f) / (b - c))) * norm;
    }
  }
  return fb;
})();

const DCT = Array.from({ length: N_MFCC }, (_, k) => {
  const s = Math.sqrt((k === 0 ? 1 : 2) / N_MELS); // type-II, norm="ortho"
  return new Float64Array(N_MELS).map((_, n) => s * Math.cos((Math.PI * k * (2 * n + 1)) / (2 * N_MELS)));
});

// ---- radix-2 FFT (in place)
const LOG2 = Math.log2(N_FFT);
const REV = new Uint16Array(N_FFT).map((_, i) => {
  let r = 0;
  for (let b = 0; b < LOG2; b++) r |= ((i >> b) & 1) << (LOG2 - 1 - b);
  return r;
});
const COS = new Float64Array(N_FFT / 2).map((_, i) => Math.cos((-2 * Math.PI * i) / N_FFT));
const SIN = new Float64Array(N_FFT / 2).map((_, i) => Math.sin((-2 * Math.PI * i) / N_FFT));

function powerSpectrum(re, im, out) {
  for (let i = 0; i < N_FFT; i++) {
    const j = REV[i];
    if (j > i) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let size = 2; size <= N_FFT; size <<= 1) {
    const half = size >> 1, step = N_FFT / size;
    for (let s = 0; s < N_FFT; s += size) {
      for (let k = 0; k < half; k++) {
        const wr = COS[k * step], wi = SIN[k * step];
        const a = s + k, b = a + half;
        const tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
      }
    }
  }
  for (let k = 0; k < N_BINS; k++) out[k] = re[k] * re[k] + im[k] * im[k];
}

function delta(feat) { // width 2, edge padding (features.delta)
  const T = feat[0].length;
  return feat.map((row) => {
    const at = (t) => row[Math.min(T - 1, Math.max(0, t))];
    return new Float64Array(T).map((_, t) => ((at(t + 1) - at(t - 1)) + 2 * (at(t + 2) - at(t - 2))) / 10);
  });
}

/** 240-d MFCC statistics for one 500 ms frame (features.mfcc_mean_vector). */
export function mfccMeanVector(frame) {
  const x = Float64Array.from(frame);
  let mean = 0;
  for (const v of x) mean += v;
  mean /= x.length;
  let peak = 0;
  for (let i = 0; i < x.length; i++) { x[i] -= mean; peak = Math.max(peak, Math.abs(x[i])); }
  for (let i = 0; i < x.length; i++) x[i] /= peak + 1e-9;

  // log-Mel (64 x T)
  const lm = Array.from({ length: N_MELS }, () => new Float64Array(N_FRAMES));
  const re = new Float64Array(N_FFT), im = new Float64Array(N_FFT), S = new Float64Array(N_BINS);
  for (let t = 0; t < N_FRAMES; t++) {
    for (let n = 0; n < N_FFT; n++) { re[n] = x[t * HOP + n] * WIN[n]; im[n] = 0; }
    powerSpectrum(re, im, S);
    for (let m = 0; m < N_MELS; m++) {
      let e = 0;
      const row = MEL_FB[m];
      for (let k = 0; k < N_BINS; k++) e += row[k] * S[k];
      lm[m][t] = Math.log(e + 1e-6);
    }
  }
  // MFCC (40 x T) + delta + delta2
  const c = DCT.map((basis) => new Float64Array(N_FRAMES).map((_, t) => {
    let s = 0;
    for (let n = 0; n < N_MELS; n++) s += basis[n] * lm[n][t];
    return s;
  }));
  const d1 = delta(c);
  const rows = [...c, ...d1, ...delta(d1)];
  const out = new Float64Array(rows.length * 2);
  rows.forEach((row, i) => {
    let m = 0;
    for (const v of row) m += v;
    m /= row.length;
    let v2 = 0;
    for (const v of row) v2 += (v - m) ** 2;
    out[i] = m;
    out[rows.length + i] = Math.sqrt(v2 / row.length);
  });
  return out;
}

/** Siren confidence c_a in [0, 1] for one 500 ms, 16 kHz frame. */
export function sirenConfidence(frame) {
  const v = mfccMeanVector(frame);
  let h = Array.from(v, (x, i) => (x - MODEL.scaler.mean[i]) / MODEL.scaler.scale[i]);
  MODEL.weights.forEach((W, l) => {
    const b = MODEL.biases[l];
    const z = Array.from(b);
    for (let i = 0; i < h.length; i++) {
      const hi = h[i];
      if (hi === 0) continue;
      const Wi = W[i];
      for (let j = 0; j < z.length; j++) z[j] += hi * Wi[j];
    }
    h = l < MODEL.weights.length - 1 ? z.map((q) => Math.max(0, q)) : z;
  });
  return 1 / (1 + Math.exp(-h[0]));
}
