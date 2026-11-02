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
import csv
import numpy as np
from scipy.optimize import minimize

from classify.trackprediction import most_specific_tag, DEFAULT_THRESHOLD
from ml_tools.interpreter import get_parent_mappings
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
    "--csv-dir",
    help="also save every table as <name>.csv in this folder",
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
multi_label = model_meta.get("hyperparams", {}).get("multi_label", False)
SEGMENT_FRAMES = 25
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
NUM_FRAMES = np.array([t["num_frames"] for t in tracks])
PLAIN = np.array([label_thresholds.get(l, DEFAULT_THRESHOLD) for l in labels])
parent_mappings = get_parent_mappings(labels) if multi_label else None
# a predicted tag can be a common parent such as mammal, these get indices after labels
tag_names = list(labels)
pattern_tags = {}


def tag_for(pattern, above):
    # most specific label over its threshold, cached as the same few patterns repeat
    if pattern not in pattern_tags:
        tag = most_specific_tag(labels, np.flatnonzero(above), parent_mappings)
        if tag not in tag_names:
            tag_names.append(tag)
        pattern_tags[pattern] = tag_names.index(tag)
    return pattern_tags[pattern]


def decide(probs, subset, c=0.0, k=2, m=1.0):
    # same as TrackPrediction: the predicted tag is the most specific label over m times
    # its threshold, else the most confident label, and it is confident if over m times
    # the threshold raised for missing frames (c=0 plain, c=1 k=2 is get_scaled_thresh)
    subset = np.asarray(subset, dtype=int)
    p = probs[subset]
    tag = p.argmax(1)
    above = p >= m * PLAIN
    has = above.any(1)
    if parent_mappings is not None and has.any():
        above_has = above[has]
        patterns = above_has @ (1 << np.arange(len(labels)))
        uniq, first, inv = np.unique(patterns, return_index=True, return_inverse=True)
        lookup = np.array(
            [tag_for(int(u), above_has[f]) for u, f in zip(uniq, first)], dtype=int
        )
        tag[has] = lookup[inv]
    missing = np.clip(SEGMENT_FRAMES - NUM_FRAMES[subset], 0, None) / (
        SEGMENT_FRAMES - 1
    )
    thresh = PLAIN + c * (missing**k)[:, None] * np.maximum(0, 0.96 - PLAIN)
    # a common parent is not a label so classify gives it no confidence
    is_label = tag < len(labels)
    idx = np.where(is_label, tag, 0)
    rows = np.arange(len(subset))
    confident = is_label & (p[rows, idx] >= m * thresh[rows, idx])
    return tag, confident, tag == Y[subset]


def tag_conf(p, tag):
    return round(p[tag], 2) if tag < len(labels) else 0

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
    subset = np.asarray(subset, dtype=int)
    if len(subset) == 0:
        return 0, 0, 0, 0
    _, confident, correct = decide(probs, subset, 1.0 if scaled else 0.0)
    ll = -np.log(probs[subset, Y[subset]] + 1e-6).mean()
    right, wrong = (confident & correct).sum(), (confident & ~correct).sum()
    return correct.mean(), int(right), int(wrong), ll


def table(name, title, header, rows):
    # print as markdown and, with --csv-dir, save the same rows as <name>.csv
    print()
    print(title)
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(str(v) for v in row) + " |")
    if args.csv_dir:
        os.makedirs(args.csv_dir, exist_ok=True)
        path = os.path.join(args.csv_dir, f"{name}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print("saved", os.path.abspath(path))


all_tracks = list(range(len(tracks)))
cv_name = f"{args.folds} fold cross validation by recording"
base = pooled(np.zeros(0), "mean (current)")
# weighting can't change a single prediction track, so fit and score on the rest
MULTI = n_preds > 1
multi_tracks = list(np.flatnonzero(MULTI))
if not args.skip_weighting:
    rows = []
    for form in FORMS:
        probs = cross_validate(
            lambda train: fit(train & MULTI, form), lambda theta: pooled(theta, form)
        )
        acc, cr, cw, ll = summarise(probs, multi_tracks)
        lacc, lcr, lcw, lll = summarise(probs, long_tracks)
        theta = fit(MULTI, form)
        names = {
            "disp power": ["(d+1)^{:.2f}"],
            "disp power + exp": ["(d+1)^{:.2f}", "exp({:.3f}·d/10)"],
            "mass power": ["(m+1)^{:.2f}"],
            "disp + mass power": ["(d+1)^{:.2f}", "(m+1)^{:.2f}"],
            "disp + mass + conf": ["(d+1)^{:.2f}", "(m+1)^{:.2f}", "conf^{:.2f}"],
            "used disp + mass power": ["(u+1)^{:.2f}", "(m+1)^{:.2f}"],
        }.get(form, [])
        formula = " · ".join(n.format(v) for n, v in zip(names, theta)) or "1"
        rows.append(
            [form, formula, round(acc * 100, 1), cr, cw, round(ll, 3)]
            + [round(lacc * 100, 1), lcr, lcw, round(lll, 3)]
        )
    table(
        "weighting",
        f"weighting on {len(multi_tracks)} tracks with > 1 prediction "
        f"({cv_name}, confident uses scaled per label thresholds):",
        ["Weighting", "Formula (fit on multi)", "Accuracy %", "Confident right"]
        + ["Confident wrong", "Log loss", "Long accuracy %", "Long confident right"]
        + ["Long confident wrong", "Long log loss"],
        rows,
    )

    probs = cross_validate(
        lambda train: fit(train & MULTI, "disp + mass power"),
        lambda theta: pooled(theta, "disp + mass power"),
    )
    scaled_c = 1.0 if scale_thresholds else 0.0
    tag_a, conf_a, _ = decide(base, long_tracks, scaled_c)
    tag_b, conf_b, _ = decide(probs, long_tracks, scaled_c)
    rows = []
    for j, i in enumerate(long_tracks):
        t = tracks[i]
        rows.append(
            [t["file"], t["id"], labels[t["y"]], len(t["P"])]
            + [tag_names[tag_a[j]], tag_conf(base[i], tag_a[j]), conf_a[j]]
            + [tag_names[tag_b[j]], tag_conf(probs[i], tag_b[j]), conf_b[j]]
            + [tag_a[j] != tag_b[j]]
        )
    table(
        "long_tracks",
        f"long tracks (current mean vs disp + mass power, {cv_name}):",
        ["File", "Track", "Truth", "Predictions", "Mean label", "Mean conf"]
        + ["Mean confident", "Weighted label", "Weighted conf", "Weighted confident"]
        + ["Changed"],
        rows,
    )

if not args.calibrate:
    sys.exit(0)

# --- calibration: per-prediction temperature from movement / mass ---
# the model is multi label with sigmoid outputs, so each label is scaled on its own
# p' = sigmoid(logit p / T), T = exp(theta . [1, phi]); changes the confidence of each
# prediction but not its argmax, the track argmax can still change once averaged


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
# predictions are saved as whole percents, so 0 means < 0.5% and 100 means > 99.5%,
# clip to the middle of those buckets so the logit is finite. classify
# --dont-round-predictions keeps 4 decimals of a percent instead
rounded = np.allclose(P_all * 100, np.round(P_all * 100))
ROUNDING_EPS = 0.0025 if rounded else 1e-6
print("predictions rounded to whole percents:", rounded)
clipped = np.clip(P_all, ROUNDING_EPS, 1 - ROUNDING_EPS)
# reverse sigmoid
LOGIT_P_all = np.log(clipped) - np.log(1 - clipped)
Y_onehot = np.eye(len(labels))[Y]
# multi label training also sets bird for chicken and weka, see get_extra_mappings
if multi_label and "bird" in labels:
    for land_bird in ("chicken", "weka"):
        if land_bird in labels:
            Y_onehot[Y == labels.index(land_bird), labels.index("bird")] = 1


def calibrated(theta, form):
    # mean of temperature scaled predictions for every track, shape (tracks, labels)
    T = np.exp(CAL_PHI[form] @ theta)[:, None]
    p = 1 / (1 + np.exp(-LOGIT_P_all / T))
    return np.add.reduceat(p, starts) / n_preds[:, None]


def bce(p):
    # binary cross entropy summed over labels for every track, as the model was trained
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(Y_onehot * np.log(p) + (1 - Y_onehot) * np.log(1 - p)).sum(1)


def cal_loss(theta, train, form, l2=0.01):
    loss = (CW * bce(calibrated(theta, form)))[train].sum() / train.sum()
    return loss + l2 * np.sum(theta[1:] ** 2)


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


def cal_row(name, formula, probs):
    _, cr, cw, _ = summarise(probs, short_tracks, scaled=True)
    _, lcr, lcw, _ = summarise(probs, short_tracks, scaled=False)
    return [name, formula, cr, cw, lcr, lcw, round(bce(probs)[short_tracks].mean(), 3)]


rows = [cal_row("none (current)", "–", base)]
cal_probs = {}
for form in CAL_FORMS:
    probs = cross_validate(
        lambda train: cal_fit(train & SHORT, form),
        lambda theta: calibrated(theta, form),
    )
    cal_probs[form] = probs
    theta = cal_fit(SHORT, form)
    formula = "T = " + " · ".join(n.format(v) for n, v in zip(CAL_NAMES[form], theta))
    rows.append(cal_row(form, formula, probs))
table(
    "calibration",
    f"calibration on short tracks ({cv_name}, scaled vs plain per label thresholds):",
    ["Calibration", "Formula (fit on short)", "Scaled confident right"]
    + ["Scaled confident wrong", "Label confident right", "Label confident wrong"]
    + ["BCE"],
    rows,
)


# the predicted tag can change with m (a chicken with only bird over the bar is called
# bird), so evaluate a grid of m rather than sorting tracks once
M_GRID = np.exp(np.linspace(np.log(0.3), np.log(3.0), 600))


def threshold_curve(probs, c=0.0, k=2):
    right, wrong = [], []
    for m in M_GRID:
        _, confident, correct = decide(probs, short_tracks, c, k, m)
        right.append((confident & correct).sum())
        wrong.append((confident & ~correct).sum())
    return np.array(right), np.array(wrong)


def best_m(curve, target):
    # grid index with the fewest confident wrong while keeping at least target right
    right, wrong = curve
    ok = np.flatnonzero(right >= target)
    return ok[np.argmin(wrong[ok])] if len(ok) else None


def wrong_at(curve, target):
    best = best_m(curve, target)
    if best is None:
        return ["", ""]
    return [int(curve[1][best]), round(M_GRID[best], 2)]


current_right = summarise(base, short_tracks, scaled=True)[1]
targets = [round(current_right * f) for f in (1.0, 0.9, 0.8, 0.6)]
# c=1 k=2 is get_scaled_thresh, c=0 plain thresholds
curves = {"none (current), scaled": threshold_curve(base, 1.0)}
curves["none, plain"] = threshold_curve(base)
for c in (0.25, 0.5):
    curves[f"none, missing^1 c={c}"] = threshold_curve(base, c, 1)
for form in CAL_FORMS:
    curves[f"{form}, plain"] = threshold_curve(cal_probs[form])
for k in (1, 2):
    for c in (0.25, 0.5, 1.0):
        curves[f"global T, missing^{k} c={c}"] = threshold_curve(
            cal_probs["global T"], c, k
        )
# displacement in the temperature and missing frames in the threshold
for form in ("T(disp)", "T(disp, mass)"):
    for c in (0.25, 0.5, 1.0):
        curves[f"{form}, missing^1 c={c}"] = threshold_curve(cal_probs[form], c, 1)
table(
    "sweep",
    "short tracks, confident wrong at matched confident right "
    "(all thresholds multiplied by m to hit the target):",
    ["Method"] + [h for r in targets for h in (f"Wrong @ {r} right", f"m @ {r} right")],
    [
        [name] + [v for r in targets for v in wrong_at(curve, r)]
        for name, curve in curves.items()
    ],
)

# per label at matched totals, the calibrated setups use the m that hits 90% of
# current confident right
label_target = targets[1]
setups = {"current": (base, 1.0, 2, 1.0)}
for name, curve_name, probs, c in (
    ("global T", "global T, plain", cal_probs["global T"], 0.0),
    ("global T linear c=0.5", "global T, missing^1 c=0.5", cal_probs["global T"], 0.5),
    ("T(disp) linear c=0.5", "T(disp), missing^1 c=0.5", cal_probs["T(disp)"], 0.5),
):
    best = best_m(curves[curve_name], label_target)
    setups[name] = (probs, c, 1, M_GRID[best] if best is not None else 1.0)

# right is counted by true label, wrong by the tag it was confidently called
right_by = {name: collections.Counter() for name in setups}
wrong_by = {name: collections.Counter() for name in setups}
for name, (probs, c, k, m) in setups.items():
    tag, confident, correct = decide(probs, short_tracks, c, k, m)
    right_by[name].update(Y[short_tracks][confident & correct])
    wrong_by[name].update(tag[confident & ~correct])
short_counts = collections.Counter(Y[short_tracks])

rows = []
for y in sorted(range(len(tag_names)), key=lambda y: -short_counts[y]):
    if short_counts[y] or any(wrong_by[s][y] for s in setups):
        rows.append(
            [tag_names[y], short_counts[y]]
            + [right_by[s][y] for s in setups]
            + [wrong_by[s][y] for s in setups]
        )
rows.append(
    ["total", len(short_tracks)]
    + [sum(right_by[s].values()) for s in setups]
    + [sum(wrong_by[s].values()) for s in setups]
)
table(
    "per_label",
    f"short tracks per label, calibrated setups matched to {label_target} confident right "
    "(right by true label, wrong by the label it was called):",
    ["Label", "Tracks"]
    + [f"Right {n} (m={m:.2f})" for n, (_, _, _, m) in setups.items()]
    + [f"Wrong {n}" for n in setups],
    rows,
)

# before is current (scaled thresholds), after is global T with plain thresholds
tag_b, conf_b, right_b = decide(base, short_tracks, 1.0)
tag_a, conf_a, _ = decide(cal_probs["global T"], short_tracks)
rows = []
for j, i in enumerate(short_tracks):
    if not right_b[j]:
        t = tracks[i]
        rows.append(
            [t["file"], t["id"], labels[t["y"]], tag_names[tag_b[j]]]
            + [tag_conf(base[i], tag_b[j]), conf_b[j], tag_names[tag_a[j]]]
            + [tag_conf(cal_probs["global T"][i], tag_a[j]), conf_a[j]]
            + [round(t["F"][:, 0].max(), 1), round(t["F"][:, 2].mean()), t["num_frames"]]
        )
table(
    "wrong_short_tracks",
    f"wrong short tracks, current vs global T with plain thresholds ({cv_name}):",
    ["File", "Track", "Truth", "Predicted before", "Conf before", "Confident before"]
    + ["Predicted after", "Conf after", "Confident after"]
    + ["Max disp", "Mean mass", "Frames"],
    rows,
)
