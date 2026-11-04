"""
plot_pokes.py  -  Morales et al. (2020)-style visualisation of 8-port maze poke data.

Input
-----
data : array, shape (N, 5), one row per poke, columns
    0  port           int 0..7, 0 = correct port, numbered consecutively round the arena
    1  trial          trial index l(t)
    2  time_in_trial  seconds since trial (cue) onset
    3  confidence     percent, 0..100
    4  water          bool, whether the poke yielded water
    (confidence given as a fraction 0..1 is detected and rescaled to percent)
min_trial : int
    Water is available on trial l(t) iff l(t) >= min_trial (as in 1(l(t) >= tau) in the
    model notes). Set STRICT = True for l(t) > min_trial instead.

Panels (paper figure in brackets)
---------------------------------
plot_raster        poke raster, trials x time, coloured by distance to correct port   [2A, 3A top]
plot_psth          cue-aligned poke rate, correct vs mean incorrect port              [2A bottom]
plot_cumulative    cumulative pokes per port vs trial                                 [2D left, 3D]
plot_distance_hist poke histogram vs distance to correct port, with uniform-surrogate
                   P<0.01 bound                                                       [2D right]
plot_polar         polar poke histogram, poke vector and memory index (MI)            [3F, 3G]
plot_mi_by_trial   MI per trial and running MI                                        [3J]
plot_confidence    model confidence per poke vs trial                                 (not in paper)
plot_session       all of the above in one figure

Usage
-----
Edit DATA_FILE and MIN_TRIAL at the bottom of this file and run it (python plot_pokes.py,
or Run in your IDE). Or import it:
    from plot_pokes import plot_session
    fig = plot_session(data, min_trial)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

PORT, TRIAL, TIME, CONF, WATER = range(5)
N_PORTS = 8
THETA = np.arange(N_PORTS) * np.pi / 4          # port angle, correct port at 0
DIST_ORDER = np.array([5, 6, 7, 0, 1, 2, 3, 4])  # ports ordered by signed distance -3..4
DIST_COLOURS = {0: "black", 1: "#8B1A1A", 2: "#D9461A", 3: "#F08C1E", 4: "#F2C12E"}
WATER_SHADE = "#DCEEFB"
WATER_EDGE = "#1F77B4"
STRICT = False   # False: available iff trial >= min_trial;  True: iff trial > min_trial


def water_available(trial, min_trial):
    trial = np.asarray(trial)
    return trial > min_trial if STRICT else trial >= min_trial


# ----------------------------------------------------------------------------- helpers
def signed_distance(port):
    """Signed circular distance to the correct port: 0,1,2,3,4,-3,-2,-1 for ports 0..7."""
    port = np.asarray(port, dtype=int)
    return np.where(port <= 4, port, port - N_PORTS)


def port_colour(port):
    return DIST_COLOURS[abs(int(signed_distance(port)))]


def unpack(data):
    """Validate, sort by (trial, time) and return the five columns."""
    data = np.asarray(data, dtype=float).reshape(-1, 5)
    data = data[np.lexsort((data[:, TIME], data[:, TRIAL]))]
    if data.size and np.nanmax(data[:, CONF]) <= 1.0:
        data[:, CONF] *= 100          # fraction -> percent
    port = data[:, PORT].astype(int)
    if port.size and (port.min() < 0 or port.max() > 7):
        raise ValueError("port must be in 0..7")
    return (port, data[:, TRIAL].astype(int), data[:, TIME],
            data[:, CONF], data[:, WATER].astype(bool))


def select_phase(data, min_trial, phase):
    """phase: 'no_water' (water unavailable), 'water' (available) or 'all'."""
    data = np.asarray(data, dtype=float).reshape(-1, 5)
    avail = water_available(data[:, TRIAL], min_trial)
    if phase == "no_water":
        return data[~avail]
    if phase == "water":
        return data[avail]
    if phase == "all":
        return data
    raise ValueError(f"unknown phase {phase!r}")


def port_counts(port):
    return np.bincount(np.asarray(port, dtype=int), minlength=N_PORTS)


def poke_vector(counts):
    """Normalised vector sum of pokes (Morales et al. Methods)."""
    counts = np.asarray(counts, dtype=float)
    n = counts / counts.sum(axis=-1, keepdims=True)
    return (n * np.cos(THETA)).sum(-1), (n * np.sin(THETA)).sum(-1)


def memory_index(counts):
    """MI = x-component of the poke vector, in [-1, 1]; +1 = all pokes in the correct port."""
    return poke_vector(counts)[0]


def uniform_surrogates(n_pokes, n_surr=500, seed=0):
    """Surrogate poke-count histograms with the same N but uniform over ports. (n_surr, 8)"""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_pokes, np.full(N_PORTS, 1 / N_PORTS), size=n_surr)


def _shade_water_trials(ax, trials, min_trial, axis="y"):
    trials = np.asarray(trials)
    avail = trials[water_available(trials, min_trial)]
    if avail.size == 0:
        return
    lo, hi = avail.min() - 0.5, trials.max() + 0.5
    span = ax.axhspan if axis == "y" else ax.axvspan
    span(lo, hi, color=WATER_SHADE, zorder=0, lw=0)


def _distance_legend(ax, **kw):
    handles = [Line2D([], [], marker="o", ls="", color=DIST_COLOURS[d], label=f"|d| = {d}")
               for d in range(5)]
    handles.append(Line2D([], [], marker="o", ls="", mfc="none", mec=WATER_EDGE, mew=1.5,
                          label="water"))
    handles.append(Patch(color=WATER_SHADE, label="water available"))
    ax.legend(handles=handles, fontsize=7, framealpha=0.9, **kw)


# ----------------------------------------------------------------------------- panels
def plot_raster(ax, data, min_trial, color_by="distance", t_max=None):
    """Poke raster. Grey bar = trial duration up to its last poke (proxy for the cue bar)."""
    port, trial, t, conf, water = unpack(data)
    trials = np.unique(trial)
    _shade_water_trials(ax, trials, min_trial)
    for tr in trials:
        ax.barh(tr, t[trial == tr].max(), height=0.8, color="0.88", zorder=1, lw=0)

    if color_by == "distance":
        ax.scatter(t, trial, c=[port_colour(p) for p in port], s=16, zorder=2)
        _distance_legend(ax, loc="upper right")
    elif color_by == "confidence":
        sc = ax.scatter(t, trial, c=conf, cmap="viridis", vmin=0, vmax=100, s=16, zorder=2)
        plt.colorbar(sc, ax=ax, label="confidence (%)", pad=0.01)
    else:
        raise ValueError("color_by must be 'distance' or 'confidence'")

    ax.scatter(t[water], trial[water], s=70, facecolors="none", edgecolors=WATER_EDGE,
               lw=1.5, zorder=3)
    ax.axvline(0, ls="--", c="0.4", lw=0.8)
    ax.set_xlim(-0.02 * (t_max or t.max()), t_max or 1.05 * t.max())
    ax.set_ylim(trials.min() - 0.6, trials.max() + 0.6)
    ax.set_xlabel("time relative to cue onset (s)")
    ax.set_ylabel("trial")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return ax


def plot_psth(ax, data, min_trial, phase="no_water", bin_s=0.5, t_max=None):
    """Cue-aligned poke rate (Hz per trial): correct port vs mean of the 7 incorrect ports."""
    port, trial, t, *_ = unpack(select_phase(data, min_trial, phase))
    if t.size == 0:
        ax.text(0.5, 0.5, "no pokes", ha="center", transform=ax.transAxes)
        return ax
    n_trials = np.unique(trial).size
    edges = np.arange(0, (t_max or t.max()) + bin_s, bin_s)
    corr, _ = np.histogram(t[port == 0], edges)
    inc, _ = np.histogram(t[port != 0], edges)
    norm = n_trials * bin_s
    ax.stairs(corr / norm, edges, color="black", lw=1.5, label="correct")
    ax.stairs(inc / (7 * norm), edges, color="0.6", lw=1.5, label="incorrect (mean)")
    ax.set_xlabel("time relative to cue onset (s)")
    ax.set_ylabel("poke rate (Hz)")
    ax.set_title(f"poke rate, {phase.replace('_', '-')} trials", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    return ax


def plot_cumulative(ax, data, min_trial):
    """Cumulative pokes per port; dashed = port at the same |distance| on the other side."""
    port, trial, *_ = unpack(data)
    trials = np.arange(trial.min(), trial.max() + 1)
    _shade_water_trials(ax, trials, min_trial, axis="x")
    for p in range(N_PORTS):
        per_trial = np.bincount(trial[port == p] - trials[0], minlength=trials.size)
        ax.plot(trials, np.cumsum(per_trial), color=port_colour(p),
                ls="--" if signed_distance(p) < 0 else "-", lw=2.2 if p == 0 else 1.2)
    ax.set_xlabel("trial")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("cumulative pokes")
    return ax


def plot_distance_hist(ax, data, min_trial, phase="no_water", n_surr=500, alpha=0.01, seed=0):
    """Pokes vs signed distance to correct port with pointwise P<alpha uniform bound."""
    port, *_ = unpack(select_phase(data, min_trial, phase))
    counts = port_counts(port)
    d = signed_distance(DIST_ORDER)
    ax.plot(d, counts[DIST_ORDER], "k-o", ms=4)
    if counts.sum():
        surr = uniform_surrogates(counts.sum(), n_surr, seed)
        ax.axhline(surr.mean(), color="0.5", ls="--", lw=0.8, label="uniform mean")
        ax.axhline(np.quantile(surr, 1 - alpha), color="m", ls="--", lw=0.8,
                   label=f"P<{alpha} bound")
        ax.legend(fontsize=7, frameon=False)
    ax.set_xticks(d)
    ax.set_xlabel("distance to correct port")
    ax.set_ylabel("pokes")
    ax.set_title(f"poke histogram, {phase.replace('_', '-')} trials", fontsize=9)
    return ax


def plot_polar(ax, data, min_trial, phase="no_water", n_surr=500, alpha=0.01, seed=0):
    """Polar poke histogram, poke vector (black arrow) and MI (red), as in Fig. 3F/G.
    `ax` must have projection='polar'."""
    port, *_ = unpack(select_phase(data, min_trial, phase))
    counts = port_counts(port)
    N = counts.sum()
    if N == 0:
        ax.set_title("no pokes", fontsize=9)
        return ax, np.nan, np.nan

    loop = np.r_[THETA, THETA[0]]
    ax.plot(loop, np.r_[counts, counts[0]], "k-o", ms=4, zorder=3)
    surr = uniform_surrogates(N, n_surr, seed)
    bound = np.quantile(surr, 1 - alpha, axis=0)
    ax.plot(loop, np.r_[bound, bound[0]], "m--", lw=0.8)

    rmax = 1.15 * max(counts.max(), bound.max())
    for p in range(N_PORTS):
        ax.scatter(THETA[p], rmax, marker="s", s=40, color=port_colour(p), zorder=4,
                   clip_on=False)

    vx, vy = poke_vector(counts)
    ax.annotate("", xy=(np.arctan2(vy, vx), np.hypot(vx, vy) * N), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    mi = vx
    ax.plot([0 if mi >= 0 else np.pi] * 2, [0, abs(mi) * N], color="red", lw=2.5)

    surr_mi = memory_index(surr)
    p_val = (1 + np.sum(surr_mi >= mi)) / (n_surr + 1)
    ax.set_ylim(0, rmax)
    ax.set_xticks(THETA)
    ax.set_xticklabels([f"{p}" for p in range(N_PORTS)], fontsize=7)
    ax.set_yticks(np.linspace(0, rmax, 4)[1:3])
    ax.set_yticklabels([f"{v:.0f}" for v in np.linspace(0, rmax, 4)[1:3]])
    ax.tick_params(axis="y", labelsize=6, labelcolor="0.4")
    ax.set_title(f"MI = {mi:.2f} (p = {p_val:.3f}), {phase.replace('_', '-')}", fontsize=9)
    return ax, mi, p_val


def plot_mi_by_trial(ax, data, min_trial, phase="no_water", n_surr=500, alpha=0.01, seed=0):
    """MI of each trial's pokes (dots) and running MI over trials so far (line)."""
    port, trial, *_ = unpack(select_phase(data, min_trial, phase))
    trials = np.unique(trial)
    if trials.size == 0:
        ax.text(0.5, 0.5, "no pokes", ha="center", transform=ax.transAxes)
        return ax
    per = np.array([port_counts(port[trial == tr]) for tr in trials])
    ax.plot(trials, memory_index(per), "o", color="0.55", ms=4, label="per trial")
    ax.plot(trials, memory_index(np.cumsum(per, 0)), "k-", lw=1.5, label="running")
    run_n = np.cumsum(per.sum(1))
    run_bound = [np.quantile(memory_index(uniform_surrogates(n, n_surr, seed)), 1 - alpha)
                 for n in run_n]
    ax.plot(trials, run_bound, "m--", lw=0.8, label=f"running P<{alpha} bound")
    ax.axhline(0, color="0.7", lw=0.6)
    ax.set_xlabel("trial")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("MI")
    ax.legend(fontsize=7, frameon=False)
    return ax


def plot_confidence(ax, data, min_trial):
    """Confidence of each poke vs trial, coloured by distance; line = mean per trial."""
    port, trial, _, conf, water = unpack(data)
    port, trial, conf = np.asarray(port), np.asarray(trial), np.asarray(conf, float)
    trials = np.unique(trial)
    _shade_water_trials(ax, trials, min_trial, axis="x")
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, trial.size)
    ax.scatter(trial + jitter, conf, c=[port_colour(p) for p in port], s=12, zorder=2)
    ax.plot(trials, [conf[trial == tr].mean() for tr in trials], "k-", lw=1, zorder=3)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("trial")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("confidence (%)")
    return ax


def plot_session(data, min_trial, color_by="distance", phase="no_water", title=None):
    """Composite figure of all panels for one session."""
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, width_ratios=[1.5, 1, 1], height_ratios=[1, 1, 0.55])
    t_max = 1.05 * np.asarray(data, dtype=float)[:, TIME].max()

    plot_raster(fig.add_subplot(gs[0:2, 0]), data, min_trial, color_by, t_max)
    plot_psth(fig.add_subplot(gs[2, 0]), data, min_trial, phase, t_max=t_max)
    plot_cumulative(fig.add_subplot(gs[0, 1]), data, min_trial)
    plot_distance_hist(fig.add_subplot(gs[0, 2]), data, min_trial, phase)
    plot_polar(fig.add_subplot(gs[1, 1], projection="polar"), data, min_trial, phase)
    plot_confidence(fig.add_subplot(gs[1, 2]), data, min_trial)
    plot_mi_by_trial(fig.add_subplot(gs[2, 1:]), data, min_trial, phase)

    rule = ">" if STRICT else "\u2265"
    fig.suptitle(title or f"water available for trial {rule} {min_trial}", fontsize=11)
    return fig


# ----------------------------------------------------------------------------- demo
def demo_data(n_trials=20, min_trial=8, seed=1):
    """Synthetic session: memory biased towards port 0 and its neighbours."""
    rng = np.random.default_rng(seed)
    prior = np.array([4, 2, 1, .5, .3, .5, 1, 2.5]); prior /= prior.sum()
    rows, found = [], False
    for tr in range(1, n_trials + 1):
        t = rng.uniform(0.3, 1.0)
        n_pokes = 1 if found else rng.integers(1, 5)
        for _ in range(n_pokes):
            p = 0 if found else rng.choice(N_PORTS, p=prior)
            w = (p == 0) and bool(water_available(tr, min_trial))
            rows.append([p, tr, t, 100 * prior[p] * rng.uniform(0.8, 1.6), w])
            t += rng.uniform(0.5, 1.5)
            if w:
                found = True
                break
    return np.array(rows), min_trial


def check(data, min_trial):
    """Warn if any water poke falls on a trial where water should be unavailable."""
    port, trial, *_, water = unpack(data)
    bad = water & ~water_available(trial, min_trial)
    if bad.any():
        print(f"warning: water on trial(s) {np.unique(trial[bad])} where it should be "
              f"unavailable; check min_trial / STRICT")


# ----------------------------------------------------------------------------- run
# # Edit these and run the file.
# MIN_TRIAL = 20
# COLOR_BY = "distance"      # "distance" or "confidence"
# PHASE = "no_water"         # pokes used for histogram, polar, poke-rate and MI panels
# SAVE_AS = None             # e.g. "session.png"; None just shows the figure

# # .npy file holding (port, trial, time_in_trial, confidence, water) per poke
# DATA_FILE = "session.npy"


def load_pokes(path):
    """Load an (N, 5) poke array from .npy. Also handles object arrays of tuples
    (e.g. np.save of a list of tuples) and structured arrays."""
    arr = np.load(path, allow_pickle=True)
    if arr.dtype.names is not None or arr.dtype == object:
        arr = np.array(arr.tolist(), dtype=float)
    return np.asarray(arr, dtype=float).reshape(-1, 5)