"""Design diagrams for the AmbuSmart report (Gantt, use case, class, sequence, state, process model)."""
import os
import shutil
import subprocess
import datetime as dt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, Ellipse

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
INK, ACC, ACC2, GREY = "#1f2a44", "#c0392b", "#2e6da4", "#8a8f98"


def dot(name, src):
    if shutil.which("dot") is None:
        print(f"Skipping {name}.png: Graphviz 'dot' not installed (macOS: brew install graphviz)")
        return
    open(f"{OUT}/{name}.dot", "w").write(src)
    subprocess.run(["dot", "-Tpng", "-Gdpi=200", f"{OUT}/{name}.dot", "-o", f"{OUT}/{name}.png"], check=True)
    os.remove(f"{OUT}/{name}.dot")


# ----------------------------------------------------------------- Gantt
def gantt():
    tasks = [
        ("Problem identification & literature survey", "2026-08-03", "2026-08-23", "Vibodharya"),
        ("Requirement analysis & SRS", "2026-08-17", "2026-08-30", "Vatsal"),
        ("System architecture & hardware design", "2026-08-31", "2026-09-20", "Vatsal"),
        ("Dataset preparation (ESC-50 + Indian sirens)", "2026-09-07", "2026-09-27", "Shreyansh"),
        ("Feature extraction & classifier training", "2026-09-21", "2026-10-18", "Shreyansh"),
        ("GPS Kalman tracker & GCC-PHAT DOA", "2026-09-28", "2026-10-18", "Pranav"),
        ("Fusion engine & Monte-Carlo simulation", "2026-10-12", "2026-11-01", "Pranav"),
        ("Dashboard simulator (React UI)", "2026-10-05", "2026-10-25", "Namitha"),
        ("Testing & performance analysis", "2026-10-26", "2026-11-12", "Namitha"),
        ("Report, IEEE paper & presentation", "2026-11-02", "2026-11-27", "Vibodharya"),
    ]
    colors = {"Vibodharya": "#2e6da4", "Vatsal": "#5b8c5a", "Shreyansh": "#c0392b",
              "Pranav": "#d68a1c", "Namitha": "#7d5ba6"}
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for i, (t, s, e, who) in enumerate(tasks):
        s, e = dt.date.fromisoformat(s), dt.date.fromisoformat(e)
        ax.barh(i, (e - s).days, left=mdates.date2num(s), color=colors[who], height=0.55)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t[0] for t in tasks])
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.setp(ax.get_xticklabels(), rotation=60, fontsize=8)
    ax.grid(axis="x", color="#ddd")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, ["Vibodharya Jampale", "Vatsal Kalra", "Shreyansh Pareek", "Pranav Anand",
                        "Namitha Karadigudda"], loc="lower left", fontsize=8, frameon=False, ncol=1)
    ax.set_title("AmbuSmart project schedule, August to November 2026", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/gantt.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------- Use case (matplotlib for real UML look)
def stick(ax, x, y, label):
    ax.add_patch(plt.Circle((x, y + 0.55), 0.17, fill=False, lw=1.5, color=INK))
    ax.plot([x, x], [y + 0.38, y - 0.15], color=INK, lw=1.5)
    ax.plot([x - 0.3, x + 0.3], [y + 0.2, y + 0.2], color=INK, lw=1.5)
    ax.plot([x, x - 0.25], [y - 0.15, y - 0.6], color=INK, lw=1.5)
    ax.plot([x, x + 0.25], [y - 0.15, y - 0.6], color=INK, lw=1.5)
    ax.text(x, y - 0.85, label, ha="center", va="top", fontsize=9, color=INK)


def use_case():
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis("off")
    ax.add_patch(FancyBboxPatch((3.0, 0.4), 8.0, 9.2, boxstyle="round,pad=0.02", fill=False, lw=1.4, color=INK))
    ax.text(7.0, 9.35, "AmbuSmart Emergency Vehicle Preemption System", ha="center", fontsize=11, weight="bold", color=INK)
    ucs = {
        "broadcast": (7.0, 8.4, "Broadcast GPS telemetry"),
        "detect": (7.0, 7.3, "Detect siren\n(MLP classifier)"),
        "doa": (7.0, 6.3, "Estimate siren bearing\n(GCC-PHAT)"),
        "track": (7.0, 5.25, "Track ambulance\n(Kalman filter)"),
        "fuse": (7.0, 4.2, "Fuse channels &\ndecide activation"),
        "preempt": (7.0, 3.15, "Run 4-phase\nsignal preemption"),
        "corridor": (7.0, 2.1, "Pre-clear green\nwave corridor"),
        "monitor": (7.0, 1.1, "Monitor events /\nmanual override"),
    }
    for k, (x, y, t) in ucs.items():
        ax.add_patch(Ellipse((x, y), 3.4, 0.85, fill=True, fc="#eef3fa", ec=ACC2, lw=1.3))
        ax.text(x, y, t, ha="center", va="center", fontsize=8.6, color=INK)
    actors = {"amb": (1.3, 7.3, "Ambulance\n(GPS unit)"), "mic": (1.3, 4.3, "Microphone\narray"),
              "op": (1.3, 1.3, "Traffic control\noperator"), "sig": (12.7, 3.6, "Traffic signal\ncontroller"),
              "ccs": (12.7, 7.0, "Central\nCoordination Server")}
    for k, (x, y, t) in actors.items():
        stick(ax, x, y, t)
    links = [("amb", "broadcast"), ("amb", "track"), ("mic", "detect"), ("mic", "doa"),
             ("op", "monitor"), ("sig", "preempt"), ("ccs", "corridor"), ("ccs", "broadcast"),
             ("sig", "corridor")]
    for a, u in links:
        ax_, ay = actors[a][:2]; ux, uy = ucs[u][:2]
        ux = ux - 1.7 if ax_ < ux else ux + 1.7
        ax.plot([ax_ + (0.35 if ax_ < 7 else -0.35), ux], [ay + 0.2, uy], color=GREY, lw=1)
    for a, b in [("detect", "fuse"), ("doa", "fuse"), ("track", "fuse")]:
        pass
    ax.annotate("", xy=(5.3, 4.2), xytext=(5.3, 7.3), arrowprops=dict(arrowstyle="-", ls="--", color=GREY))
    ax.text(4.9, 5.8, "«include»", rotation=90, fontsize=8, color=GREY, ha="center")
    ax.annotate("", xy=(7, 3.6), xytext=(7, 3.8), arrowprops=dict(arrowstyle="->", ls="--", color=GREY))
    ax.text(7.95, 3.68, "«include»", fontsize=7.5, color=GREY)
    ax.annotate("", xy=(7, 2.55), xytext=(7, 2.72), arrowprops=dict(arrowstyle="<-", ls="--", color=GREY))
    ax.text(7.95, 2.6, "«extend»", fontsize=7.5, color=GREY)
    fig.tight_layout()
    fig.savefig(f"{OUT}/use_case.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------- Class diagram
def class_diagram():
    dot("class_diagram", r'''
digraph G { rankdir=TB; nodesep=0.5; ranksep=0.6;
 node [shape=record, fontname="Helvetica", fontsize=10, style=filled, fillcolor="#f4f7fb", color="#1f2a44"];
 edge [fontname="Helvetica", fontsize=9, color="#555555"];
 Feature [label="{FeatureExtractor|+ sr: int = 16000\l+ n_fft: int = 512\l+ n_mels: int = 64\l+ n_mfcc: int = 40\l|+ logmel(x): ndarray\l+ mfcc_stack(lm): ndarray\l+ mfcc_mean_vector(x): 240-d\l}"];
 CNN [label="{SirenClassifier|- scaler: StandardScaler\l- mlp: 240-128-64-1 (ReLU)\l|+ fit(X, y)\l+ predict(frame): c_a\l}"];
 DOA [label="{BearingEstimator|- mics: 4 x 2 array\l- pairs: 6\l|+ gcc_phat(a, b): tau\l+ estimate_bearing(sig): theta_a\l}"];
 GPS [label="{GPSTracker|- kf: CVKalman\l- last_packet_t: float\l|+ on_packet(pkt): bool\l+ verify_hmac(pkt): bool\l+ extrapolate(t): (dist, bearing)\l+ available(): bool\l}"];
 KF [label="{CVKalman|- x: [px, py, vx, vy]\l- P: 4x4 covariance\l|+ predict(dt)\l+ update(z)\l}"];
 Fusion [label="{FusionEngine|- thr_gps: 0.60\l- thr_acoustic: 0.85 (N=3)\l- thr_fallback: 0.90 (N=5)\l- bearing_tol: 30 deg\l|+ step(c_a, theta_a, gps): Decision\l}"];
 SM [label="{PreemptionController|- state: IDLE / PREPARE / GREEN /\l           RESUME / SUPPRESS\l- queue: PriorityQueue (max 4)\l|+ trigger(axis, tta)\l+ tick(dt)\l+ confirm_passage()\l}"];
 Relay [label="{SignalInterface|+ assert_relay(axis)\l+ release()\l}"];
 Logger [label="{EventLogger|+ log(event)\l+ flush_to_nv()\l}"];
 CNN -> Feature [label=" uses", arrowhead=vee, style=dashed];
 GPS -> KF [dir=back, arrowtail=diamond, label=" 1"];
 Fusion -> CNN [arrowhead=vee, label=" c_a"];
 Fusion -> DOA [arrowhead=vee, label=" theta_a"];
 Fusion -> GPS [arrowhead=vee, label=" position"];
 SM -> Fusion [dir=back, arrowtail=vee, label=" Decision"];
 SM -> Relay [arrowhead=vee];
 SM -> Logger [arrowhead=vee, style=dashed];
 {rank=same; CNN; DOA; GPS}
}''')


# ----------------------------------------------------------------- State diagram
def state_diagram():
    dot("state_diagram", r'''
digraph S { rankdir=LR; nodesep=0.45; ranksep=0.55;
 node [shape=box, style="rounded,filled", fillcolor="#eef3fa", color="#2e6da4", fontname="Helvetica", fontsize=10];
 edge [fontname="Helvetica", fontsize=8.5, color="#444444"];
 start [shape=circle, label="", width=0.18, style=filled, fillcolor="#1f2a44"];
 IDLE [label="IDLE\n(normal cycle)"];
 PREP [label="PHASE 1\nPREPARE\nP = min(8 s, 0.8 TTA)"];
 GREEN [label="PHASE 2\nPRIORITY GREEN\n30 s, +10 s up to 60 s"];
 RES [label="PHASE 3\nRESUME CYCLE\nfrom paused position"];
 SUP [label="PHASE 4\nSUPPRESS\n60 s, dual-channel only"];
 SAFE [label="FAIL-SAFE\nnormal operation", fillcolor="#fbeeee", color="#c0392b"];
 start -> IDLE;
 IDLE -> PREP [label="fusion = ACTIVATE"];
 PREP -> GREEN [label="cross axes red"];
 GREEN -> GREEN [label="no passage:\nextend 10 s"];
 GREEN -> RES [label="passage confirmed\n(c_a<0.30 x3 or GPS stop-line)"];
 GREEN -> RES [label="timeout 60 s\n(log event)"];
 RES -> SUP;
 SUP -> IDLE [label="60 s elapsed"];
 SUP -> PREP [label="both channels\nconfirm"];
 IDLE -> SAFE [label="watchdog 2 s", style=dashed];
 GREEN -> SAFE [style=dashed];
 SAFE -> IDLE [label="reboot / restore state", style=dashed];
}''')


# ----------------------------------------------------------------- Sequence diagram
def sequence():
    lanes = ["Ambulance\nGPS unit", "Central\nServer", "Microphone\narray", "Classifier +\nGCC-PHAT", "Fusion\nengine",
             "Preemption\ncontroller", "Signal\ncontroller"]
    msgs = [
        (0, 1, "UDP packet + HMAC (1 Hz)"),
        (1, 4, "relay telemetry"),
        (4, 4, "verify HMAC, Kalman update"),
        (2, 3, "500 ms audio frame (every 250 ms)"),
        (3, 3, "MFCC features, MLP inference"),
        (3, 4, "c_a, theta_a"),
        (4, 4, "dual-path threshold check"),
        (4, 5, "ACTIVATE(axis, TTA)"),
        (5, 6, "Phase 1: cross axes amber -> red"),
        (5, 6, "Phase 2: assert green on ambulance axis"),
        (3, 5, "c_a < 0.30 x3 (siren passed)"),
        (5, 6, "Phase 3: resume cycle from pause"),
        (5, 1, "event log / alert"),
    ]
    fig, ax = plt.subplots(figsize=(12, 7.4))
    n = len(lanes)
    xs = [0.8 + i * 1.75 for i in range(n)]
    top = len(msgs) + 1.2
    ax.set_xlim(0, xs[-1] + 0.9); ax.set_ylim(-0.6, top + 0.9); ax.axis("off")
    for x, l in zip(xs, lanes):
        ax.add_patch(FancyBboxPatch((x - 0.72, top - 0.05), 1.44, 0.85, boxstyle="round,pad=0.02",
                                    fc="#eef3fa", ec=ACC2))
        ax.text(x, top + 0.37, l, ha="center", va="center", fontsize=8.5, color=INK)
        ax.plot([x, x], [-0.4, top - 0.05], color=GREY, ls="--", lw=0.9)
    for i, (a, b, t) in enumerate(msgs):
        y = top - 0.8 - i
        if a == b:
            x = xs[a]
            ax.add_patch(plt.Rectangle((x - 0.06, y - 0.35), 0.12, 0.5, fc="#dfe7f2", ec=ACC2, lw=0.8))
            ax.plot([x + 0.06, x + 0.32, x + 0.32], [y + 0.08, y + 0.08, y - 0.22], color=INK, lw=1)
            ax.annotate("", xy=(x + 0.07, y - 0.22), xytext=(x + 0.33, y - 0.22),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1))
            ax.text(x + 0.38, y - 0.1, t, fontsize=8, va="center", color=INK)
        else:
            ax.annotate("", xy=(xs[b], y), xytext=(xs[a], y),
                        arrowprops=dict(arrowstyle="-|>", color=ACC if b in (5, 6) else INK, lw=1.1))
            ax.text((xs[a] + xs[b]) / 2, y + 0.12, t, ha="center", fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/sequence.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------- Process model
def process_model():
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.4); ax.axis("off")
    stages = ["Requirements\n& survey", "Design", "Implement\nincrement", "Test &\nevaluate", "Review with\nguide"]
    for i, s in enumerate(stages):
        x = 0.4 + i * 2.15
        ax.add_patch(FancyBboxPatch((x, 1.4), 1.7, 1.0, boxstyle="round,pad=0.04", fc="#eef3fa", ec=ACC2))
        ax.text(x + 0.85, 1.9, s, ha="center", va="center", fontsize=9.5, color=INK)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + 2.15, 1.9), xytext=(x + 1.75, 1.9), arrowprops=dict(arrowstyle="-|>", color=INK))
    ax.annotate("", xy=(4.75, 1.35), xytext=(9.85, 1.35),
                arrowprops=dict(arrowstyle="-|>", color=ACC, connectionstyle="arc3,rad=-0.35", lw=1.4))
    ax.text(7.3, 0.25, "next increment: I1 acoustic classifier  ->  I2 GPS + DOA  ->  I3 fusion + simulator  ->  I4 dashboard",
            ha="center", fontsize=9, color=ACC)
    fig.tight_layout()
    fig.savefig(f"{OUT}/process_model.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    gantt(); use_case(); class_diagram(); state_diagram(); sequence(); process_model()
    print(sorted(os.listdir(OUT)))
