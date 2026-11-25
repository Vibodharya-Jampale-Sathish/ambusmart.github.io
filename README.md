# AmbuSmart — AI Emergency Vehicle Preemption

3rd-semester mini project · Dept. of AI&ML · Ramaiah Institute of Technology · 2026-27
Guide: **Ms. Priya M**, Teaching Assistant

| Name | USN |
|---|---|
| Vibodharya Jampale | 1MS25AI079 |
| Vatsal Kalra | 1MS25AI077 |
| Shreyansh Pareek | 1MS25AI067 |
| Pranav Anand | 1MS25AI050 |
| Namitha Karadigudda | 1MS25AI047 |

## Folder layout
```
ai/          Python: features, dataset, training, bearing + GPS, fusion simulation, plots, tests
  results/   trained model (.pkl) and all result numbers (.json) – already included
  figures/   every figure used in the report and paper
dashboard/   React + Vite + Tailwind traffic-signal simulator
data/        put ESC-50 here only if you want to retrain (see data/README.md)
docs/        project report, IEEE paper, presentation script
```

## Setup (once)
Open this folder in VS Code (File → Open Folder), then in the terminal:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Select the `.venv` interpreter when VS Code asks.

## Run
Everything below runs from the `ai` folder (the VS Code launch configs already do this).

| Step | Command | Needs ESC-50? | Time |
|---|---|---|---|
| Tests (22) | `python -m pytest -v` | No | seconds |
| Bearing + GPS experiments | `python doa_gps.py` | No | ~40 s |
| Fusion simulation | `python fusion_sim.py` | No | ~25 s |
| Design diagrams | `python diagrams.py` | No (needs Graphviz `dot`) | seconds |
| Build dataset | `python dataset.py` | Yes | ~1 min |
| Train 5 classifiers | `python train.py` | Yes (after dataset) | ~1.5 min |
| Result figures | `python plots.py` | Yes (for one figure) | seconds |

The trained model and results are already in `ai/results/`, so tests and the simulation work straight away.

## Dashboard
```bash
cd dashboard
npm install
npm run dev
```
Open the address it prints (usually http://localhost:5173) and press **Start Simulation**.

**Live Microphone** tab: press **Start Listening**, allow the microphone, and play a siren near it
(a siren video, or the built-in Wail / Yelp / Hi-Lo test sirens). The trained MLP runs in the browser
on every 500 ms of audio (every 250 ms). The signal turns green when confidence stays above the threshold
for 3 frames, then goes amber and back to red once the siren has been gone for 3 frames (minimum 5 s green).
No audio leaves the device. Microphone access needs `localhost` or an `https://` site.

If you retrain the model, run `python export_web_model.py` (from `ai/`) to update the weights used by the dashboard.

Deploy: `npm run build` and upload `dashboard/dist/` to any static host (Netlify, Vercel, GitHub Pages).

## Key results (held-out test set)
- MLP (128-64) siren classifier: **95.0 % accuracy**, ROC-AUC 0.983 (best of 5 models)
- Fusion simulation (9,000 episodes): **99.96 %** timely activation, **1.4 %** false activation
  (single-frame sound-only detection: 99.6 % false activation)
