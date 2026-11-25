// AmbuSmart - live microphone -> siren confidence, one score every 250 ms on the last 500 ms of audio
// (the same frame / hop the classifier and fusion engine were designed for).
import { useEffect, useRef, useState } from 'react';
import { FRAME_LEN, SR, sirenConfidence } from './classifier.js';

const HOP_MS = 250;
const SILENCE_RMS = 5e-4; // below this the frame is treated as silence (score 0)

// Windowed-sinc resampler to 16 kHz (only used if the browser will not open a 16 kHz AudioContext).
function resampleTo16k(x, rate) {
  if (rate === SR) return x;
  const ratio = rate / SR;
  const fc = 0.45 / ratio; // anti-alias cutoff, cycles per input sample
  const half = Math.ceil(16 * ratio);
  const out = new Float32Array(FRAME_LEN);
  for (let i = 0; i < FRAME_LEN; i++) {
    const t = i * ratio;
    const c = Math.floor(t);
    let s = 0, wsum = 0;
    for (let k = c - half + 1; k <= c + half; k++) {
      if (k < 0 || k >= x.length) continue;
      const d = t - k;
      const sinc = d === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * d) / (Math.PI * d);
      const w = 0.5 + 0.5 * Math.cos((Math.PI * d) / half); // Hann window
      s += x[k] * sinc * w;
      wsum += sinc * w;
    }
    out[i] = wsum ? s / wsum : 0;
  }
  return out;
}

async function openContext(stream) {
  // Prefer a native 16 kHz context (browser resamples the mic); fall back to the device rate.
  let ctx;
  try {
    ctx = new AudioContext({ sampleRate: SR });
    return { ctx, source: ctx.createMediaStreamSource(stream) };
  } catch {
    ctx?.close();
    ctx = new AudioContext();
    return { ctx, source: ctx.createMediaStreamSource(stream) };
  }
}

/**
 * onScore(c, rms) is called every 250 ms while listening.
 * Returns { listening, error, start, stop }.
 */
export function useMicSiren(onScore) {
  const [listening, setListening] = useState(false);
  const [error, setError] = useState(null);
  const cb = useRef(onScore);
  cb.current = onScore;
  const rt = useRef(null);

  const stop = () => {
    const r = rt.current;
    rt.current = null;
    if (!r) return;
    clearInterval(r.timer);
    r.stream.getTracks().forEach((t) => t.stop());
    r.ctx.close();
    setListening(false);
  };

  const start = async () => {
    if (rt.current) return;
    setError(null);
    let stream, ctx;
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Microphone access needs a secure page (https:// or localhost).');
      }
      // Browser voice processing would suppress a siren as "noise" or cancel it as "echo": turn it off.
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
      });
      const opened = await openContext(stream);
      ctx = opened.ctx;
      const { source } = opened;
      await ctx.audioWorklet.addModule(`${import.meta.env.BASE_URL}mic-capture-worklet.js`);
      const node = new AudioWorkletNode(ctx, 'mic-capture');
      const mute = ctx.createGain();
      mute.gain.value = 0;
      source.connect(node).connect(mute).connect(ctx.destination);

      const rate = ctx.sampleRate;
      const ring = new Float32Array(Math.ceil((FRAME_LEN / SR) * rate));
      let w = 0, filled = 0;
      node.port.onmessage = ({ data }) => {
        for (let i = 0; i < data.length; i++) {
          ring[w] = data[i];
          w = (w + 1) % ring.length;
        }
        filled = Math.min(ring.length, filled + data.length);
      };

      const timer = setInterval(() => {
        if (filled < ring.length) return;
        const raw = new Float32Array(ring.length);
        raw.set(ring.subarray(w));
        raw.set(ring.subarray(0, w), ring.length - w);
        const frame = resampleTo16k(raw, rate);
        let e = 0;
        for (const v of frame) e += v * v;
        const rms = Math.sqrt(e / frame.length);
        cb.current(rms < SILENCE_RMS ? 0 : sirenConfidence(frame), rms);
      }, HOP_MS);

      if (ctx.state === 'suspended') await ctx.resume();
      rt.current = { ctx, stream, timer };
      setListening(true);
    } catch (e) {
      setError(e.name === 'NotAllowedError' ? 'Microphone permission was denied.' : e.message);
      stream?.getTracks().forEach((t) => t.stop());
      ctx?.close();
    }
  };

  useEffect(() => stop, []);
  return { listening, error, start, stop };
}

// ---------------------------------------------------------------- built-in test siren (plays through the speakers)
const PATTERNS = {
  wail: { lfo: 'sine', rate: 0.25, center: 650, depth: 150 }, // 500-800 Hz slow sweep
  yelp: { lfo: 'triangle', rate: 4, center: 1000, depth: 500 }, // 500-1500 Hz, ~4 Hz
  hilo: { lfo: 'square', rate: 0.75, center: 550, depth: 110 }, // 440 / 660 Hz alternation
};

export function playTestSiren(kind = 'wail') {
  const p = PATTERNS[kind];
  const ctx = new AudioContext();
  const osc = ctx.createOscillator();
  osc.type = 'square';
  osc.frequency.value = p.center;
  const lfo = ctx.createOscillator();
  lfo.type = p.lfo;
  lfo.frequency.value = p.rate;
  const depth = ctx.createGain();
  depth.gain.value = p.depth;
  lfo.connect(depth).connect(osc.frequency);
  const lp = ctx.createBiquadFilter();
  lp.type = 'lowpass';
  lp.frequency.value = 4000;
  const vol = ctx.createGain();
  vol.gain.value = 0.12;
  osc.connect(lp).connect(vol).connect(ctx.destination);
  osc.start();
  lfo.start();
  return () => ctx.close();
}
