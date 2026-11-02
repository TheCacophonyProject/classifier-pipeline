"""Fit a per-prediction weighting w = exp(theta . phi) that pools a track's
predictions into one label, using human tags as ground truth.

With --calibrate also fit a per-prediction temperature that changes how
confident each prediction is (works on single prediction tracks too).

usage: python3 fit_weights.py <classified dir> [original dir with human tags]
           [-c config.toml] [--model-json saved_model.json] [--long 5] [--calibrate]

classify keeps the original metadata (including human tags) when it writes to
--output-dir, so the original dir is only needed if the tags were overwritten
"""

import argparse
import sys
import json
import glob
import os
import collections
import numpy as np
from scipy.optimize import minimize

from classify.trackprediction import get_scaled_thresh, DEFAULT_THRESHOLD
from config.config import Config
from track.track import Track

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", help="folder of classified metadata")
parser.add_argument(
    "orig_dir",
    nargs="?",
    help="folder of original metadata with human tags, defaults to run_dir",
)
parser.add_argument(
    "--model-json",
    default="/etc/cacophony/models/server-v1.2/default/saved_model.json",
    help="model metadata with per label thresholds",
)
parser.add_argument(
    "-c", "--config-file", help="classifier config, used for tag precedence"
)
parser.add_argument(
    "--long",
    type=int,
    default=5,
    help="tracks with at least this many predictions are reported as long",
)
parser.add_argument(
    "--calibrate",
    action="store_true",
    help="also fit a confidence calibration from movement and mass",
)
parser.add_argument(
    "--folds",
    type=int,
    default=5,
    help="cross validation folds, tracks from one recording stay in the same fold",
)
parser.add_argument(
    "--skip-weighting",
    action="store_true",
    help="skip the weighting comparison and long track listing",
)
args = parser.parse_args()

model_meta = json.load(open(args.model_json))
label_thresholds = model_meta.get("thresholds", {})
scale_thresholds = model_meta.get("scale_thresholds", False)
tag_precedence = Config.load_from_file(args.config_file).build.tag_precedence


def centre(q):
    return np.array([q["x"] + q["width"] / 2, q["y"] + q["height"] / 2])


def human_tags(path):
    try:
        d = json.load(open(path))
    except Exception:
        return {}
    tags = {}
    for t in d.get("Tracks") or d.get("tracks", []):
        # same choice of human tag as Track.load_track_meta
        tag = Track.get_best_human_tag(
            t.get("tags"), tag_precedence, min_confidence=0.8
        )
        if tag:
            tags[t["id"]] = tag["what"]
    return tags


tracks = []
labels = None
no_mass = 0
for f in sorted(glob.glob(os.path.join(args.run_dir, "**/*.txt"), recursive=True)):
    rel = os.path.relpath(f, args.run_dir)
    truth = human_tags(os.path.join(args.orig_dir or args.run_dir, rel))
    try:
        d = json.load(open(f))
    except Exception:
        continue
    # prefer "Tracks" for positions, classify writes predictions into "tracks" but
    # files may only have one of the keys (e.g. after audio-training fixtracks.py)
    upper = {x["id"]: x for x in d.get("Tracks") or []}
    lower = {x["id"]: x for x in d.get("tracks") or []}
    for track_id in list(upper) + [i for i in lower if i not in upper]:
        if track_id not in truth:
            continue
        t = lower.get(track_id) or upper[track_id]
        preds = [p for p in t.get("predictions") or [] if p.get("predictions")]
        if not preds and track_id in upper:
            t = upper[track_id]
            preds = [p for p in t.get("predictions") or [] if p.get("predictions")]
        positions = (upper.get(track_id) or {}).get("positions") or t.get("positions")
        if not preds or not positions:
            continue
        # some metadata has no mass, either missing or all zero
        if not any(q.get("mass") for q in positions):
            no_mass += 1
            continue
        p = preds[0]
        if labels is None:
            labels = list(p["all_class_confidences"].keys())
        if truth[t["id"]] not in labels:
            continue
        pos = {q.get("frame_number", q.get("order")): q for q in positions}
        P, feats = [], []
        num_frames = 0
        for s in p["predictions"]:
            fr = s["frames"]
            lo, hi = min(fr), max(fr)
            rng = [
                pos[o]
                for o in range(lo, hi + 1)
                if o in pos and not pos[o].get("blank", False)
            ]
            used = [pos[o] for o in fr if o in pos and not pos[o].get("blank", False)]
            if len(rng) == 0:
                continue
            pts = np.array([centre(q) for q in rng])
            upts = np.array([centre(q) for q in used]) if used else pts[:1]
            max_disp = np.linalg.norm(pts - pts[0], axis=1).max()
            used_disp = np.linalg.norm(np.diff(upts, axis=0), axis=1).sum()
            # mean mass over used frames, independent of how many frames the segment got
            mean_mass = np.mean([q["mass"] for q in used]) if used else 0
            pr = np.array(s["prediction"], dtype=float) / 100
            P.append(pr)
            feats.append((max_disp, used_disp, mean_mass, pr.max(), len(fr)))
            num_frames += len(fr)
        if P:
            tracks.append(
                dict(
                    file=rel,
                    id=t["id"],
                    y=labels.index(truth[t["id"]]),
                    P=np.array(P),
                    F=np.array(feats),
                    num_frames=num_frames,
                )
            )


def threshold_for(t, label_index, scaled=scale_thresholds):
    # same as TrackPrediction: per label threshold, raised for tracks with few frames
    threshold = label_thresholds.get(labels[label_index], DEFAULT_THRESHOLD)
    if scaled:
        threshold = get_scaled_thresh(threshold, t["num_frames"])
    return threshold


def is_confident(t, p, scaled=scale_thresholds):
    return p.max() >= threshold_for(t, p.argmax(), scaled)


long_tracks = [i for i, t in enumerate(tracks) if len(t["P"]) >= args.long]
print(f"{len(tracks)} labelled tracks, {sum(len(t['P']) for t in tracks)} predictions")
print("skipped tracks with no mass:", no_mass)
print("labels:", collections.Counter(labels[t["y"]] for t in tracks).most_common())
print("tracks with >1 prediction:", sum(len(t["P"]) > 1 for t in tracks))
print(f"long tracks (>= {args.long} predictions):", len(long_tracks))
print("scale thresholds:", scale_thresholds)

# candidate weighting forms, each a function of the raw features
FORMS = {
    "mean (current)": lambda F: np.zeros((len(F), 0)),
    "disp power": lambda F: np.c_[np.log1p(F[:, 0])],
    "disp power + exp": lambda F: np.c_[np.log1p(F[:, 0]), F[:, 0] / 10],
    "mass power": lambda F: np.c_[np.log1p(F[:, 2])],
    "disp + mass power": lambda F: np.c_[np.log1p(F[:, 0]), np.log1p(F[:, 2])],
    "disp + mass + conf": lambda F: np.c_[
        np.log1p(F[:, 0]), np.log1p(F[:, 2]), np.log(F[:, 3] + 1e-3)
    ],
    "used disp + mass power": lambda F: np.c_[np.log1p(F[:, 1]), np.log1p(F[:, 2])],
}

counts = collections.Counter(t["y"] for t in tracks)
class_w = {k: len(tracks) / (len(counts) * v) for k, v in counts.items()}

# every prediction stacked into flat arrays, each track is a contiguous block
# starting at starts[i], so pooling is a reduceat rather than a loop over tracks
P_all = np.concatenate([t["P"] for t in tracks])
F_all = np.concatenate([t["F"] for t in tracks])
n_preds = np.array([len(t["P"]) for t in tracks])
starts = np.r_[0, np.cumsum(n_preds)[:-1]]
seg = np.repeat(np.arange(len(tracks)), n_preds)
Y = np.array([t["y"] for t in tracks])
CW = np.array([class_w[y] for y in Y])
ALL = np.ones(len(tracks), dtype=bool)

# grouped k-fold, all tracks from one recording land in the same fold
files = sorted({t["file"] for t in tracks})
file_fold = dict(
    zip(np.random.default_rng(0).permutation(files), np.arange(len(files)) % args.folds)
)
fold = np.array([file_fold[t["file"]] for t in tracks])

PHI = {form: f(F_all) for form, f in FORMS.items()}


def pooled(theta, form):
    # pooled probabilities for every track, shape (tracks, labels)
    z = PHI[form] @ theta
    w = np.exp(z - np.maximum.reduceat(z, starts)[seg])
    return (
        np.add.reduceat(w[:, None] * P_all, starts)
        / np.add.reduceat(w, starts)[:, None]
    )


def weighted_nll(p, train):
    ll = -np.log(p[np.arange(len(Y)), Y] + 1e-6)
    return (CW * ll)[train].sum() / train.sum()


def loss(theta, train, form, l2=0.01):
    return weighted_nll(pooled(theta, form), train) + l2 * np.sum(theta**2)


def fit(train, form):
    n = PHI[form].shape[1]
    if n == 0:
        return np.zeros(0)
    return minimize(loss, np.zeros(n), args=(train, form), method="L-BFGS-B").x


def cross_validate(fit_fn, predict_fn):
    # fit on the other folds and predict this one, so every track is scored unseen
    probs = np.zeros((len(tracks), len(labels)))
    for k in range(args.folds):
        test = fold == k
        probs[test] = predict_fn(fit_fn(~test))[test]
    return probs


def summarise(probs, subset, scaled=scale_thresholds):
    # accuracy, confident right, confident wrong, log loss over the subset of tracks
    right = conf_right = conf_wrong = 0
    ll = 0
    for i in subset:
        t, p = tracks[i], probs[i]
        correct = p.argmax() == t["y"]
        right += correct
        if is_confident(t, p, scaled):
            conf_right += correct
            conf_wrong += not correct
        ll -= np.log(p[t["y"]] + 1e-6)
    n = max(len(subset), 1)
    return right / n, conf_right, conf_wrong, ll / n


all_tracks = list(range(len(tracks)))
cv_name = f"{args.folds} fold cross validation by recording"
base = pooled(np.zeros(0), "mean (current)")
if not args.skip_weighting:
    print()
    print(f"weighting ({cv_name}, confident uses scaled per label thresholds):")
    print(
        "| Weighting | Formula (fit on all) | Accuracy | Confident right | Confident wrong | Log loss "
        f"| Long accuracy | Long confident right | Long confident wrong | Long log loss |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for form in FORMS:
        probs = cross_validate(
            lambda train: fit(train, form), lambda theta: pooled(theta, form)
        )
        acc, cr, cw, ll = summarise(probs, all_tracks)
        lacc, lcr, lcw, lll = summarise(probs, long_tracks)
        theta = fit(ALL, form)
        names = {
            "disp power": ["(d+1)^{:.2f}"],
            "disp power + exp": ["(d+1)^{:.2f}", "exp({:.3f}·d/10)"],
            "mass power": ["(m+1)^{:.2f}"],
            "disp + mass power": ["(d+1)^{:.2f}", "(m+1)^{:.2f}"],
            "disp + mass + conf": ["(d+1)^{:.2f}", "(m+1)^{:.2f}", "conf^{:.2f}"],
            "used disp + mass power": ["(u+1)^{:.2f}", "(m+1)^{:.2f}"],
        }.get(form, [])
        formula = " · ".join(n.format(v) for n, v in zip(names, theta)) or "1"
        print(
            f"| {form} | {formula} | {acc:.1%} | {cr} | {cw} | {ll:.3f} "
            f"| {lacc:.1%} | {lcr} | {lcw} | {lll:.3f} |"
        )

    print()
    print(f"long tracks (current mean vs disp + mass power, {cv_name}):")
    probs = cross_validate(
        lambda train: fit(train, "disp + mass power"),
        lambda theta: pooled(theta, "disp + mass power"),
    )
    for i in long_tracks:
        t = tracks[i]
        a = base[i]
        b = probs[i]
        flag = "" if a.argmax() == b.argmax() else "  <-- changed"
        print(
            f"  {t['file']:55s} {t['id']:>9} truth={labels[t['y']]:14s} n={len(t['P']):3d} "
            f"mean={labels[a.argmax()]}({a.max():.2f}{'' if is_confident(t, a) else ' unconf'}) "
            f"weighted={labels[b.argmax()]}({b.max():.2f}{'' if is_confident(t, b) else ' unconf'}){flag}"
        )

if not args.calibrate:
    sys.exit(0)

# --- calibration: per-prediction temperature from movement / mass ---
# p' = softmax(log p / T), T = exp(theta . [1, phi]); changes the confidence of each
# prediction but not its argmax, the track argmax can still change once averaged
SEGMENT_FRAMES = 25


def missing(F):
    # fraction of the segment's frames that were missing, 0 for a full segment
    return np.clip(SEGMENT_FRAMES - F[:, 4], 0, None) / SEGMENT_FRAMES


def missing_sq(F):
    # same scaling factor as get_scaled_thresh, 1 for a single frame
    return (np.clip(SEGMENT_FRAMES - F[:, 4], 0, None) / (SEGMENT_FRAMES - 1)) ** 2


CAL_FORMS = {
    "global T": lambda F: np.ones((len(F), 1)),
    "T(disp)": lambda F: np.c_[np.ones(len(F)), np.log1p(F[:, 0])],
    "T(mass)": lambda F: np.c_[np.ones(len(F)), np.log1p(F[:, 2])],
    "T(disp, mass)": lambda F: np.c_[
        np.ones(len(F)), np.log1p(F[:, 0]), np.log1p(F[:, 2])
    ],
    "T(missing)": lambda F: np.c_[np.ones(len(F)), missing(F)],
    "T(missing²)": lambda F: np.c_[np.ones(len(F)), missing_sq(F)],
    "T(disp, mass, missing)": lambda F: np.c_[
        np.ones(len(F)), np.log1p(F[:, 0]), np.log1p(F[:, 2]), missing(F)
    ],
}
CAL_NAMES = {
    "global T": ["e^{:.2f}"],
    "T(disp)": ["e^{:.2f}", "(d+1)^{:.2f}"],
    "T(mass)": ["e^{:.2f}", "(m+1)^{:.2f}"],
    "T(disp, mass)": ["e^{:.2f}", "(d+1)^{:.2f}", "(m+1)^{:.2f}"],
    "T(missing)": ["e^{:.2f}", "e^({:.2f}·missing)"],
    "T(missing²)": ["e^{:.2f}", "e^({:.2f}·missing²)"],
    "T(disp, mass, missing)": [
        "e^{:.2f}",
        "(d+1)^{:.2f}",
        "(m+1)^{:.2f}",
        "e^({:.2f}·missing)",
    ],
}
CAL_PHI = {form: f(F_all) for form, f in CAL_FORMS.items()}
LOG_P_all = np.log(P_all + 1e-4)


def calibrated(theta, form):
    # mean of temperature scaled predictions for every track, shape (tracks, labels)
    T = np.exp(CAL_PHI[form] @ theta)[:, None]
    z = LOG_P_all / T
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    return np.add.reduceat(p, starts) / n_preds[:, None]


def cal_loss(theta, train, form, l2=0.01):
    return weighted_nll(calibrated(theta, form), train) + l2 * np.sum(theta[1:] ** 2)


def cal_fit(train, form):
    n = CAL_PHI[form].shape[1]
    return minimize(cal_loss, np.zeros(n), args=(train, form), method="L-BFGS-B").x


# calibrate only short tracks, a single prediction from less than a full segment
SHORT = np.array(
    [len(t["P"]) == 1 and t["num_frames"] < SEGMENT_FRAMES for t in tracks]
)
short_tracks = list(np.flatnonzero(SHORT))
print()
print(f"short tracks (1 prediction, < {SEGMENT_FRAMES} frames):", len(short_tracks))
if not short_tracks:
    sys.exit(0)
print(f"calibration on short tracks ({cv_name}, scaled vs plain per label thresholds):")
print(
    "| Calibration | Formula (fit on short) | Scaled confident right | Scaled confident wrong "
    "| Label confident right | Label confident wrong | Log loss |"
)
print("|---|---|---|---|---|---|---|")


def cal_row(name, formula, probs):
    _, cr, cw, ll = summarise(probs, short_tracks, scaled=True)
    _, lcr, lcw, _ = summarise(probs, short_tracks, scaled=False)
    print(f"| {name} | {formula} | {cr} | {cw} | {lcr} | {lcw} | {ll:.3f} |")


cal_row("none (current)", "–", base)
cal_probs = {}
for form in CAL_FORMS:
    probs = cross_validate(
        lambda train: cal_fit(train & SHORT, form),
        lambda theta: calibrated(theta, form),
    )
    cal_probs[form] = probs
    theta = cal_fit(SHORT, form)
    formula = "T = " + " · ".join(n.format(v) for n, v in zip(CAL_NAMES[form], theta))
    cal_row(form, formula, probs)


def missing_threshold(t, label_index, c, k=2, max_threshold=0.96):
    # get_scaled_thresh with a strength c and power k, c=1 k=2 is the current scaling
    threshold = label_thresholds.get(labels[label_index], DEFAULT_THRESHOLD)
    missing = max(0, SEGMENT_FRAMES - t["num_frames"]) / (SEGMENT_FRAMES - 1)
    return threshold + c * missing**k * max(0, max_threshold - threshold)


def threshold_curve(probs, thresh_fn):
    # confident means p.max() >= m * threshold, so sorting by p.max() / threshold gives
    # confident right / wrong for every threshold multiplier m at once
    ratio, correct = [], []
    for i in short_tracks:
        t, p = tracks[i], probs[i]
        ratio.append(p.max() / thresh_fn(t, p.argmax()))
        correct.append(p.argmax() == t["y"])
    order = np.argsort(ratio)[::-1]
    correct = np.array(correct)[order]
    return np.array(ratio)[order], np.cumsum(correct), np.cumsum(~correct)


def wrong_at(curve, target):
    # fewest confident wrong while keeping at least target confident right
    ratio, right, wrong = curve
    k = np.searchsorted(right, target)
    if k >= len(right):
        return "–"
    return f"{wrong[k]} (×{ratio[k]:.2f})"


current_right = summarise(base, short_tracks, scaled=True)[1]
targets = [round(current_right * f) for f in (1.0, 0.9, 0.8, 0.6)]
print()
print(
    "short tracks, confident wrong at matched confident right "
    "(all thresholds multiplied by ×m to hit the target):"
)
print("| Method | " + " | ".join(f"Wrong @ {r} right" for r in targets) + " |")
print("|---|" + "---|" * len(targets))
scaled_fn = lambda t, i: threshold_for(t, i, True)
plain_fn = lambda t, i: threshold_for(t, i, False)
curves = {"none (current), scaled": threshold_curve(base, scaled_fn)}
curves["none, plain"] = threshold_curve(base, plain_fn)
for form in CAL_FORMS:
    curves[f"{form}, plain"] = threshold_curve(cal_probs[form], plain_fn)
# global T with thresholds raised by missing frames, c=1 k=2 is get_scaled_thresh
for k in (1, 2):
    for c in (0.25, 0.5, 1.0):
        curves[f"global T, missing^{k} c={c}"] = threshold_curve(
            cal_probs["global T"], lambda t, i: missing_threshold(t, i, c, k)
        )
for name, curve in curves.items():
    print(f"| {name} | " + " | ".join(wrong_at(curve, r) for r in targets) + " |")

# per label at matched totals, the calibrated setups use the m that hits 90% of
# current confident right
label_target = targets[1]
linear_fn = lambda t, i: missing_threshold(t, i, 0.25, 1)
setups = {"current": (base, scaled_fn, 1.0)}
for name, probs, fn in (
    ("global T", cal_probs["global T"], plain_fn),
    ("global T linear c=0.25", cal_probs["global T"], linear_fn),
):
    ratio, right, _ = threshold_curve(probs, fn)
    setups[name] = (
        probs,
        fn,
        ratio[min(np.searchsorted(right, label_target), len(ratio) - 1)],
    )

# right is counted by true label, wrong by the label it was confidently called
right_by = {name: collections.Counter() for name in setups}
wrong_by = {name: collections.Counter() for name in setups}
for name, (probs, fn, m) in setups.items():
    for i in short_tracks:
        t, p = tracks[i], probs[i]
        if p.max() >= m * fn(t, p.argmax()):
            if p.argmax() == t["y"]:
                right_by[name][t["y"]] += 1
            else:
                wrong_by[name][p.argmax()] += 1
short_counts = collections.Counter(tracks[i]["y"] for i in short_tracks)

print()
print(
    f"short tracks per label, calibrated setups matched to {label_target} confident right "
    "(right by true label, wrong by the label it was called):"
)
print(
    "| Label | Tracks | "
    + " | ".join(f"Right {n} (×{m:.2f})" for n, (_, _, m) in setups.items())
    + " | "
    + " | ".join(f"Wrong {n}" for n in setups)
    + " |"
)
print("|---|---|" + "---|" * (2 * len(setups)))
for y in sorted(range(len(labels)), key=lambda y: -short_counts[y]):
    n = short_counts[y]
    if not n and not any(wrong_by[s][y] for s in setups):
        continue
    print(
        f"| {labels[y]} | {n} | "
        + " | ".join(str(right_by[s][y]) for s in setups)
        + " | "
        + " | ".join(str(wrong_by[s][y]) for s in setups)
        + " |"
    )
print(
    f"| total | {len(short_tracks)} | "
    + " | ".join(str(sum(right_by[s].values())) for s in setups)
    + " | "
    + " | ".join(str(sum(wrong_by[s].values())) for s in setups)
    + " |"
)

print()
print(f"wrong short tracks, before / after T(disp, mass) ({cv_name}):")
for i in short_tracks:
    t, b = tracks[i], base[i]
    if b.argmax() != t["y"]:
        a = cal_probs["T(disp, mass)"][i]
        print(
            f"  {t['file']:50s} truth={labels[t['y']]:14s} pred={labels[b.argmax()]:14s} "
            f"thresh={threshold_for(t, b.argmax()):.2f} {b.max():.2f} -> {a.max():.2f} "
            f"disp={t['F'][:, 0].max():.1f} mass={t['F'][:, 2].mean():.0f} frames={t['num_frames']}"
        )
