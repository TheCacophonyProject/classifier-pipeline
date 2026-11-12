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
from scipy.special import expit

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
    "--score",
    choices=["sigmoid", "softmax", "family-softmax"],
    default="sigmoid",
    help="how --calibrate rescales each prediction: sigmoid(logit p / T) per label, "
    "softmax(log p / T) over all labels, or softmax leaving out each label's parents "
    "and children (e.g. bird for chicken)",
)
parser.add_argument(
    "--ignore-mistakes",
    nargs="*",
    default=[],
    metavar="TRUTH:CALLED",
    help="mistakes that don't matter, e.g. chicken:bird, count as neither right nor wrong "
    "when tuning and scoring",
)
parser.add_argument(
    "--wrong-cost",
    type=float,
    nargs="+",
    default=[1.0],
    help="also tune short track thresholds per label to maximise "
    "confident right - cost x confident wrong, one setup per cost",
)
parser.add_argument(
    "--label-keep",
    type=float,
    default=0.9,
    help="per label threshold tuning keeps at least this fraction of each label's "
    "current confident right on short tracks",
)
parser.add_argument(
    "--tune-dir",
    help="classified metadata (e.g. the validation split) to fit and tune everything on, "
    "run_dir is then only scored, instead of cross validating within run_dir",
)
parser.add_argument(
    "--save-fit",
    help="save everything fitted and tuned (weighting, calibration and short track "
    "thresholds) to this json file",
)
parser.add_argument(
    "--load-fit",
    help="use the values saved by --save-fit instead of fitting, run_dir is only scored",
)
parser.add_argument(
    "--save-thresholds",
    help="save the tuned short track thresholds to this json file",
)
parser.add_argument(
    "--fold-by",
    choices=["device", "recording"],
    default="device",
    help="cross validation groups, device keeps every recording from one device and "
    "station (deviceId + stationId, else device name) in the same fold",
)
parser.add_argument(
    "--csv-dir",
    help="also save every table as <name>.csv in this folder",
)
parser.add_argument(
    "--short-stats",
    action="store_true",
    help="only print per label stats (mass, other tracks at the same time, short tracks "
    "in the recording) for short tracks the model got right and wrong",
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
FPS = 9
# a track shorter than this many seconds counts towards a recording's short tracks
SHORT_SECONDS = 3
# another track's box within this many pixels in the same frame, likely the same animal
NEARBY_PIXELS = 10
# mass under this share of the box area suggests something wrong with the background
LOW_FILL = 0.1
# thermal frames, the tracker ignores a 1 pixel border (edge-pixels)
FRAME_WIDTH, FRAME_HEIGHT, EDGE_PIXELS = 160, 120, 1
tag_precedence = Config.load_from_file(args.config_file).build.tag_precedence


def centre(q):
    return np.array([q["x"] + q["width"] / 2, q["y"] + q["height"] / 2])


def track_span(t):
    # start and end in seconds, from the metadata or else the frame numbers
    if t.get("start") is not None and t.get("end") is not None:
        return float(t["start"]), float(t["end"])
    frames = [q.get("frame_number", q.get("order")) for q in t.get("positions") or []]
    frames = [f for f in frames if f is not None]
    if not frames:
        return 0.0, 0.0
    return min(frames) / FPS, (max(frames) + 1) / FPS


def on_edge(q, margin=EDGE_PIXELS):
    # box touches the edge of the usable frame, so the animal may be partly out of view
    return (
        q["x"] <= margin
        or q["y"] <= margin
        or q["x"] + q["width"] >= FRAME_WIDTH - margin
        or q["y"] + q["height"] >= FRAME_HEIGHT - margin
    )


def frame_of(q):
    return q.get("frame_number", q.get("order"))


def box_gap(a, b):
    # pixels between two boxes, 0 when they touch or overlap
    gap_x = max(a["x"], b["x"]) - min(a["x"] + a["width"], b["x"] + b["width"])
    gap_y = max(a["y"], b["y"]) - min(a["y"] + a["height"], b["y"] + b["height"])
    return max(gap_x, gap_y, 0)


def max_jump(positions):
    # largest move of the box centre between consecutive frames, in box lengths
    seen = sorted(
        (q for q in positions if not q.get("blank", False) and frame_of(q) is not None),
        key=frame_of,
    )
    jumps = [
        np.linalg.norm(centre(b) - centre(a)) / max(np.sqrt(a["width"] * a["height"]), 1)
        for a, b in zip(seen, seen[1:])
    ]
    return max(jumps, default=0.0)


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


def device_group(d, rel):
    # deviceId + stationId when the metadata has them, else the device name from the
    # file name, e.g. 1724532-20240316-121651-Tawh114.txt
    device = d.get("deviceId") or d.get("deviceName")
    if device is None:
        parts = os.path.splitext(os.path.basename(rel))[0].split("-", 3)
        device = parts[3] if len(parts) == 4 else rel
    station = d.get("stationId")
    return f"{device}/{station}" if station is not None else str(device)


tracks = []
labels = None
no_mass = 0


def load_dir(run_dir, orig_dir, tune):
    global labels, no_mass
    for f in sorted(glob.glob(os.path.join(run_dir, "**/*.txt"), recursive=True)):
        load_file(f, os.path.relpath(f, run_dir), orig_dir or run_dir, tune)


def load_file(f, rel, orig_dir, tune):
    global labels, no_mass
    truth = human_tags(os.path.join(orig_dir, rel))
    try:
        d = json.load(open(f))
    except Exception:
        return
    # prefer "Tracks" for positions, classify writes predictions into "tracks" but
    # files may only have one of the keys (e.g. after audio-training fixtracks.py)
    upper = {x["id"]: x for x in d.get("Tracks") or []}
    lower = {x["id"]: x for x in d.get("tracks") or []}
    # every track in the recording, labelled or not, for other tracks at the same time
    spans = {i: track_span({**lower.get(i, {}), **upper.get(i, {})}) for i in {*upper, *lower}}
    n_short = sum(end - start < SHORT_SECONDS for start, end in spans.values())
    # boxes per frame for every track, to find another track right next to this one
    boxes = {
        i: {
            frame_of(q): q
            for q in ({**lower.get(i, {}), **upper.get(i, {})}.get("positions") or [])
            if not q.get("blank", False) and frame_of(q) is not None
        }
        for i in spans
    }
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
            # eucl distances
            max_disp = np.linalg.norm(pts - pts[0], axis=1).max()
            used_disp = np.linalg.norm(np.diff(upts, axis=0), axis=1).sum()
            # mean mass over used frames, independent of how many frames the segment got
            mean_mass = np.mean([q["mass"] for q in used]) if used else 0
            # movement in box lengths rather than pixels, so distance to the camera matters less
            rel_disp = max_disp / np.sqrt(np.mean([q["width"] * q["height"] for q in rng]))
            edge_frac = np.mean([on_edge(q) for q in used or rng])
            # frames in the segment's range the tracker lost (blank or missing)
            blank_frac = 1 - len(rng) / (hi - lo + 1)
            masses = np.array([q["mass"] for q in used or rng], dtype=float)
            mass_cv = masses.std() / masses.mean() if masses.mean() > 0 else 0
            pr = np.array(s["prediction"], dtype=float) / 100
            P.append(pr)
            feats.append(
                (max_disp, used_disp, mean_mass, pr.max(), len(fr))
                + (rel_disp, edge_frac, blank_frac, mass_cv)
            )
            num_frames += len(fr)
        if P:
            tracks.append(
                dict(
                    tune=tune,
                    group=device_group(d, rel),
                    file=rel,
                    id=t["id"],
                    y=labels.index(truth[t["id"]]),
                    P=np.array(P),
                    F=np.array(feats),
                    num_frames=num_frames,
                    # recording context
                    track_mass=np.mean(
                        [q.get("mass") or 0 for q in positions if not q.get("blank", False)]
                        or [0]
                    ),
                    seconds=spans[track_id][1] - spans[track_id][0],
                    concurrent=sum(
                        other != track_id
                        and start < spans[track_id][1]
                        and spans[track_id][0] < end
                        for other, (start, end) in spans.items()
                    ),
                    recording_short=n_short,
                    nearby=any(
                        frame in other_boxes and box_gap(q, other_boxes[frame]) <= NEARBY_PIXELS
                        for other, other_boxes in boxes.items()
                        if other != track_id
                        for frame, q in boxes[track_id].items()
                    ),
                    edge_1=any(on_edge(q, 1) for q in boxes[track_id].values()),
                    edge_5=any(on_edge(q, 5) for q in boxes[track_id].values()),
                    max_jump=max_jump(positions),
                    blank_share=np.mean([q.get("blank", False) for q in positions]),
                    # share of the box the mass fills, very low suggests a bad background
                    fill=np.mean(
                        [
                            (q.get("mass") or 0) / max(q["width"] * q["height"], 1)
                            for q in boxes[track_id].values()
                        ]
                        or [0]
                    ),
                )
            )


load_dir(args.run_dir, args.orig_dir, False)
if args.tune_dir:
    load_dir(args.tune_dir, None, True)
# fit and tune on the tune dir when given, everything is scored on run_dir only
TUNE = np.array([t["tune"] for t in tracks])
EVAL = ~TUNE
# with --load-fit and no --tune-dir nothing is fit, so no tracks are fit on, with both the
# saved values are used where present and anything missing is fit on the tune dir
FIT = TUNE if args.tune_dir or args.load_fit else ~TUNE
LOADED = json.load(open(args.load_fit)) if args.load_fit else None
# start from the loaded values so --save-fit keeps anything this run skips (e.g. weighting)
SAVED = {
    kind: dict((LOADED or {}).get(kind, {}))
    for kind in ("weighting", "calibration", "thresholds")
}


def is_loaded(kind, name):
    return LOADED is not None and name in LOADED.get(kind, {})


def fitted(kind, name, fit_fn):
    # fit on the tune set (or run_dir), or with --load-fit return the saved value
    if is_loaded(kind, name):
        value = np.array(LOADED[kind][name], dtype=float)
    elif LOADED is not None and not args.tune_dir:
        sys.exit(
            f"{args.load_fit} has no saved {kind} for {name}, pass --tune-dir to fit it "
            "or refit with --save-fit"
        )
    else:
        value = fit_fn()
    SAVED[kind][name] = np.asarray(value, dtype=float).tolist()
    return value


def finish():
    if args.save_fit:
        SAVED["settings"] = settings()
        with open(args.save_fit, "w") as f:
            json.dump(SAVED, f, indent=4)
        print("saved", os.path.abspath(args.save_fit))
    sys.exit(0)


def is_rounded(P):
    return np.allclose(P * 100, np.round(P * 100))


if args.tune_dir and all(is_rounded(t["P"]) for t in tracks if not t["tune"]):
    # run_dir has whole percent predictions, round the tune dir the same way so the
    # thresholds and temperatures are tuned on what they are scored on
    print("run dir predictions are whole percents, rounding the tune dir to match")
    for t in tracks:
        if t["tune"]:
            t["P"] = np.round(t["P"] * 100) / 100
            t["F"][:, 3] = t["P"].max(1)

long_tracks = [
    i for i, t in enumerate(tracks) if len(t["P"]) >= args.long and not t["tune"]
]
if args.tune_dir:
    print(f"tune dir {args.tune_dir}: {TUNE.sum()} labelled tracks to fit and tune on")
    print(f"run dir {args.run_dir}: {EVAL.sum()} labelled tracks to score")
print(f"{len(tracks)} labelled tracks, {sum(len(t['P']) for t in tracks)} predictions")
print("skipped tracks with no mass:", no_mass)
print("labels:", collections.Counter(labels[t["y"]] for t in tracks).most_common())
print("tracks with >1 prediction:", sum(len(t["P"]) > 1 for t in tracks))
print(f"long tracks (>= {args.long} predictions):", len(long_tracks))
print("scale thresholds:", scale_thresholds)


def settings():
    # what the fitted values depend on, checked when they are loaded
    return {
        "labels": labels,
        "multi_label": multi_label,
        "score": args.score,
        "label_keep": args.label_keep,
        "segment_frames": SEGMENT_FRAMES,
        "model_json": args.model_json,
        "model_thresholds": label_thresholds,
        "fit_on": args.tune_dir or args.run_dir,
    }


if LOADED is not None:
    saved, now = LOADED.get("settings", {}), settings()
    for key in ("labels", "multi_label", "score"):
        if saved.get(key) != now[key]:
            sys.exit(
                f"{args.load_fit} was fit with {key}={saved.get(key)} but this run has "
                f"{now[key]}, refit with --save-fit"
            )
    for key in ("label_keep", "segment_frames", "model_thresholds"):
        if saved.get(key) != now[key]:
            print(f"warning: {args.load_fit} was fit with a different {key}")
    print(f"loaded fitted values from {args.load_fit}, fit on {saved.get('fit_on')}")

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
    # r = displacement in box lengths, e = share of frames on the frame edge,
    # b = share of frames lost by the tracker, v = mass std / mean
    "rel disp + mass power": lambda F: np.c_[np.log1p(F[:, 5]), np.log1p(F[:, 2])],
    "disp + mass + edge": lambda F: np.c_[np.log1p(F[:, 0]), np.log1p(F[:, 2]), F[:, 6]],
    "disp + mass + tracking": lambda F: np.c_[
        np.log1p(F[:, 0]), np.log1p(F[:, 2]), F[:, 7], F[:, 8]
    ],
    "disp + mass + edge + tracking": lambda F: np.c_[
        np.log1p(F[:, 0]), np.log1p(F[:, 2]), F[:, 6], F[:, 7], F[:, 8]
    ],
}

# class weights from the tracks that are fit on
counts = collections.Counter(t["y"] for t, f in zip(tracks, FIT) if f)
class_w = {k: sum(counts.values()) / (len(counts) * v) for k, v in counts.items()}

# every prediction stacked into flat arrays, each track is a contiguous block
# starting at starts[i], so pooling is a reduceat rather than a loop over tracks
P_all = np.concatenate([t["P"] for t in tracks])
F_all = np.concatenate([t["F"] for t in tracks])
n_preds = np.array([len(t["P"]) for t in tracks])
starts = np.r_[0, np.cumsum(n_preds)[:-1]]
seg = np.repeat(np.arange(len(tracks)), n_preds)
Y = np.array([t["y"] for t in tracks])
CW = np.array([class_w.get(y, 1.0) for y in Y])
ALL = np.ones(len(tracks), dtype=bool)
NUM_FRAMES = np.array([t["num_frames"] for t in tracks])
PLAIN = np.array([label_thresholds.get(l, DEFAULT_THRESHOLD) for l in labels])
parent_mappings = get_parent_mappings(labels) if multi_label else None
IGNORED = []
for pair in args.ignore_mistakes:
    truth, _, called = pair.partition(":")
    if truth not in labels or called not in labels:
        sys.exit(f"--ignore-mistakes {pair}: expected TRUTH:CALLED with labels from {labels}")
    IGNORED.append((labels.index(truth), labels.index(called)))
if IGNORED:
    print("ignoring mistakes:", ", ".join(args.ignore_mistakes))
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


def decide(probs, subset, c=0.0, k=2, m=1.0, thresholds=None):
    # same as TrackPrediction: the predicted tag is the most specific label over m times
    # its threshold, else the most confident label, and it is confident if over m times
    # the threshold raised for missing frames (c=0 plain, c=1 k=2 is get_scaled_thresh)
    plain = PLAIN if thresholds is None else thresholds
    subset = np.asarray(subset, dtype=int)
    p = probs[subset]
    tag = p.argmax(1)
    above = p >= m * plain
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
    thresh = plain + c * (missing**k)[:, None] * np.maximum(0, 0.96 - plain)
    # a common parent is not a label so classify gives it no confidence
    is_label = tag < len(labels)
    idx = np.where(is_label, tag, 0)
    rows = np.arange(len(subset))
    confident = is_label & (p[rows, idx] >= m * thresh[rows, idx])
    if IGNORED:
        # mistakes that don't matter (e.g. a chicken called bird) count as neither
        skip = np.zeros(len(subset), dtype=bool)
        for truth, called in IGNORED:
            skip |= (Y[subset] == truth) & (tag == called)
        confident = confident & ~skip
    return tag, confident, tag == Y[subset]


def tag_conf(p, tag):
    return round(p[tag], 2) if tag < len(labels) else 0


if args.tune_dir:
    # one split, fit on the tune dir (fold 1) and score run_dir (fold 0)
    fold = np.where(TUNE, 1, 0)
    EVAL_FOLDS = [0]
else:
    # grouped k-fold, all tracks from one device + station (or one recording) share a
    # fold, largest groups first into the emptiest fold so folds stay a similar size
    group_key = "group" if args.fold_by == "device" else "file"
    group_sizes = collections.Counter(t[group_key] for t in tracks)
    fold_sizes = np.zeros(args.folds)
    group_fold = {}
    for g, n in sorted(group_sizes.items(), key=lambda x: (-x[1], x[0])):
        group_fold[g] = int(fold_sizes.argmin())
        fold_sizes[group_fold[g]] += n
    fold = np.array([group_fold[t[group_key]] for t in tracks])
    EVAL_FOLDS = range(args.folds)
    print(
        f"folds by {args.fold_by}: {len(group_sizes)} groups, "
        f"tracks per fold {fold_sizes.astype(int).tolist()}"
    )

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


def cross_validate(fit_fn, predict_fn, kind=None, name=None):
    if is_loaded(kind, name):
        # nothing to fit, predict every track with the saved values
        return predict_fn(np.array(LOADED[kind][name], dtype=float))
    # fit on the other folds and predict this one, so every track is scored unseen
    probs = np.zeros((len(tracks), len(labels)))
    for k in EVAL_FOLDS:
        test = fold == k
        predicted = predict_fn(fit_fn(~test))
        probs[test] = predicted[test]
        if args.tune_dir:
            # the tune tracks too, thresholds are tuned on them after this fit
            probs[FIT] = predicted[FIT]
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


if args.tune_dir:
    cv_name = f"fit on {args.tune_dir}, scored on {args.run_dir}"
else:
    cv_name = f"{args.folds} fold cross validation by {args.fold_by}"
base = pooled(np.zeros(0), "mean (current)")

if args.short_stats:
    # short tracks scored like classify with the current thresholds, then per true label
    # compare the ones confidently right with the ones confidently wrong
    short_eval = np.flatnonzero(
        (n_preds == 1) & (NUM_FRAMES < SEGMENT_FRAMES) & EVAL
    )
    tag, confident, correct = decide(base, short_eval, 1.0 if scale_thresholds else 0.0)
    tag_score = [tag_conf(base[i], g) for i, g in zip(short_eval, tag)]
    groups = {"right": confident & correct, "wrong": confident & ~correct}

    def pct(values):
        return round(100 * float(np.mean(values)), 1)

    def stats(idx):
        if len(idx) == 0:
            return [0] + [""] * 14
        ts = [tracks[short_eval[j]] for j in idx]
        mass = np.array([t["track_mass"] for t in ts])
        return [
            len(ts),
            round(float(np.median([tag_score[j] for j in idx])), 2),
            round(float(np.median(mass)), 1),
            pct(mass <= 5),
            round(float(np.median([t["num_frames"] for t in ts])), 1),
            round(float(np.median([t["seconds"] for t in ts])), 2),
            pct([t["concurrent"] > 0 for t in ts]),
            pct([t["nearby"] for t in ts]),
            pct([t["recording_short"] > 2 for t in ts]),
            pct([t["edge_1"] for t in ts]),
            pct([t["edge_5"] for t in ts]),
            pct([t["max_jump"] > 1 for t in ts]),
            round(100 * float(np.mean([t["blank_share"] for t in ts])), 1),
            round(100 * float(np.median([t["fill"] for t in ts])), 1),
            pct([t["fill"] < LOW_FILL for t in ts]),
        ]

    y_short = Y[short_eval]
    rows = []
    for l in sorted(range(len(labels)), key=lambda l: -(y_short == l).sum()):
        for name, mask in groups.items():
            if (y_short == l).any():
                rows.append([labels[l], name] + stats(np.flatnonzero(mask & (y_short == l))))
    for name, mask in groups.items():
        rows.append(["all", name] + stats(np.flatnonzero(mask)))
    stat_header = (
        ["Tracks", "Median confidence", "Median mass", "Mass <= 5 %"]
        + ["Median frames", "Median seconds", "Other track at the same time %"]
        + [f"Track within {NEARBY_PIXELS}px at the same time %"]
        + [f"Recording has > 2 tracks under {SHORT_SECONDS}s %"]
        + ["On edge (1px) %", "On edge (5px) %", "Jump > 1 box %", "Blank frames %"]
        + ["Median fill %", f"Fill < {LOW_FILL:.0%} %"]
    )
    table(
        "short_stats",
        "short tracks (1 prediction, < 25 frames) confidently right vs confidently wrong "
        "with the current thresholds, per true label:",
        ["Label", "Group"] + stat_header,
        rows,
    )
    # the same split by the label the model called, which is what a rule can use at
    # run time, so right and wrong here share a prediction and differ only in the truth
    rows = []
    for l in sorted(range(len(labels)), key=lambda l: -(confident & (tag == l)).sum()):
        if (confident & (tag == l)).any():
            for name, mask in groups.items():
                rows.append([labels[l], name] + stats(np.flatnonzero(mask & (tag == l))))
    for name, mask in groups.items():
        rows.append(["all", name] + stats(np.flatnonzero(mask)))
    table(
        "short_stats_called",
        "short tracks confidently right vs confidently wrong with the current thresholds, "
        "per label the model called:",
        ["Called", "Group"] + stat_header,
        rows,
    )
    finish()
# weighting can't change a single prediction track, so fit and score on the rest
MULTI = n_preds > 1
multi_tracks = list(np.flatnonzero(MULTI & EVAL))
if not args.skip_weighting:
    rows = []
    for form in FORMS:
        probs = cross_validate(
            lambda train: fit(train & MULTI, form),
            lambda theta: pooled(theta, form),
            "weighting",
            form,
        )
        acc, cr, cw, ll = summarise(probs, multi_tracks)
        lacc, lcr, lcw, lll = summarise(probs, long_tracks)
        theta = fitted("weighting", form, lambda: fit(MULTI & FIT, form))
        names = {
            "disp power": ["(d+1)^{:.2f}"],
            "disp power + exp": ["(d+1)^{:.2f}", "exp({:.3f}·d/10)"],
            "mass power": ["(m+1)^{:.2f}"],
            "disp + mass power": ["(d+1)^{:.2f}", "(m+1)^{:.2f}"],
            "disp + mass + conf": ["(d+1)^{:.2f}", "(m+1)^{:.2f}", "conf^{:.2f}"],
            "used disp + mass power": ["(u+1)^{:.2f}", "(m+1)^{:.2f}"],
            "rel disp + mass power": ["(r+1)^{:.2f}", "(m+1)^{:.2f}"],
            "disp + mass + edge": ["(d+1)^{:.2f}", "(m+1)^{:.2f}", "e^({:.2f}·e)"],
            "disp + mass + tracking": ["(d+1)^{:.2f}", "(m+1)^{:.2f}"]
            + ["e^({:.2f}·b)", "e^({:.2f}·v)"],
            "disp + mass + edge + tracking": ["(d+1)^{:.2f}", "(m+1)^{:.2f}"]
            + ["e^({:.2f}·e)", "e^({:.2f}·b)", "e^({:.2f}·v)"],
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
        "weighting",
        "disp + mass power",
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
    finish()

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
    "T(rel disp)": lambda F: np.c_[np.ones(len(F)), np.log1p(F[:, 5])],
    "T(edge)": lambda F: np.c_[np.ones(len(F)), F[:, 6]],
    "T(tracking)": lambda F: np.c_[np.ones(len(F)), F[:, 7], F[:, 8]],
    "T(edge, tracking)": lambda F: np.c_[np.ones(len(F)), F[:, 6], F[:, 7], F[:, 8]],
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
    "T(rel disp)": ["e^{:.2f}", "(r+1)^{:.2f}"],
    "T(edge)": ["e^{:.2f}", "e^({:.2f}·e)"],
    "T(tracking)": ["e^{:.2f}", "e^({:.2f}·b)", "e^({:.2f}·v)"],
    "T(edge, tracking)": ["e^{:.2f}", "e^({:.2f}·e)", "e^({:.2f}·b)", "e^({:.2f}·v)"],
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
LOG_P_all = np.log(clipped)


def ancestors(label):
    found = []
    while parent_mappings and isinstance(parent_mappings.get(label), tuple):
        label = parent_mappings[label][0]
        found.append(label)
    return found


# OUTSIDE[j, l] is 1 when label j counts against label l in family-softmax, so every
# label except l's parents and children (a chicken's bird is not competition)
OUTSIDE = np.ones((len(labels), len(labels)))
for l, name in enumerate(labels):
    for j, other in enumerate(labels):
        if j != l and (other in ancestors(name) or name in ancestors(other)):
            OUTSIDE[j, l] = 0
Y_onehot = np.eye(len(labels))[Y]
# multi label training also sets bird for chicken and weka, see get_extra_mappings
if multi_label and "bird" in labels:
    for land_bird in ("chicken", "weka"):
        if land_bird in labels:
            Y_onehot[Y == labels.index(land_bird), labels.index("bird")] = 1


def calibrated(theta, form):
    # mean of temperature scaled predictions for every track, shape (tracks, labels)
    T = np.exp(CAL_PHI[form] @ theta)[:, None]
    if args.score == "sigmoid":
        p = expit(LOGIT_P_all / T)
    else:
        z = LOG_P_all / T
        e = np.exp(z - z.max(1, keepdims=True))
        if args.score == "softmax":
            p = e / e.sum(1, keepdims=True)
        else:
            p = e / (e @ OUTSIDE)
    return np.add.reduceat(p, starts) / n_preds[:, None]


def bce(p):
    # binary cross entropy summed over labels for every track, as the model was trained
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(Y_onehot * np.log(p) + (1 - Y_onehot) * np.log(1 - p)).sum(1)


def cal_loss(theta, train, form, l2=0.01):
    p = calibrated(theta, form)
    if args.score == "softmax":
        # a distribution over labels, so the log loss of the true label
        loss = weighted_nll(p, train)
    else:
        loss = (CW * bce(p))[train].sum() / train.sum()
    return loss + l2 * np.sum(theta[1:] ** 2)


def cal_fit(train, form):
    n = CAL_PHI[form].shape[1]
    return minimize(cal_loss, np.zeros(n), args=(train, form), method="L-BFGS-B").x


# calibrate only short tracks, a single prediction from less than a full segment
SHORT = np.array(
    [len(t["P"]) == 1 and t["num_frames"] < SEGMENT_FRAMES for t in tracks]
)
# short_tracks are scored, short_fit are fit and tuned on (the same without --tune-dir)
short_tracks = list(np.flatnonzero(SHORT & EVAL))
short_fit = list(np.flatnonzero(SHORT & FIT))
print()
print(f"short tracks (1 prediction, < {SEGMENT_FRAMES} frames):", len(short_tracks))
if args.tune_dir:
    print("short tracks to tune on:", len(short_fit))
if not short_tracks:
    finish()


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
        "calibration",
        form,
    )
    cal_probs[form] = probs
    theta = fitted("calibration", form, lambda: cal_fit(SHORT & FIT, form))
    formula = "T = " + " · ".join(n.format(v) for n, v in zip(CAL_NAMES[form], theta))
    rows.append(cal_row(form, formula, probs))
table(
    "calibration",
    f"calibration on short tracks, {args.score} score "
    f"({cv_name}, scaled vs plain per label thresholds):",
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

# per label thresholds tuned for short tracks, so a method is judged only on how it
# orders tracks within each label and not on how a single m treats each threshold
CANDIDATES = np.r_[np.linspace(0.02, 0.99, 98), 1.01]


def targets_for(subset):
    # label_keep of each label's current confident right within subset
    _, confident, correct = decide(base, subset, 1.0)
    right = np.bincount(
        Y[np.asarray(subset)][confident & correct], minlength=len(labels)
    )
    return right * args.label_keep


# children first, so bird is tuned knowing which chickens are already called chicken
tune_order = sorted(range(len(labels)), key=lambda l: -len(ancestors(labels[l])))


def label_counts(probs, thresholds, c=0.0, subset=None):
    # c raises each tuned threshold for missing frames, linearly as missing^1
    subset = np.asarray(short_tracks if subset is None else subset, dtype=int)
    tag, confident, correct = decide(probs, subset, c, 1, thresholds=thresholds)
    right = np.bincount(Y[subset][confident & correct], minlength=len(labels))
    wrong = np.bincount(tag[confident & ~correct], minlength=len(tag_names))
    return right, wrong[: len(labels)]


def tune_thresholds(probs, c=0.0, subset=None, passes=2, cost=None):
    # each label keeps at least label_keep of its current confident right with the fewest
    # confident wrong called as it, or as many right as it can if that is out of reach.
    # with cost, each label instead maximises confident right - cost x confident wrong
    subset = short_tracks if subset is None else subset
    label_targets = targets_for(subset)
    thresholds = PLAIN.copy()
    for _ in range(passes):
        for l in tune_order:
            best = None
            for cand in CANDIDATES:
                thresholds[l] = cand
                right, wrong = label_counts(probs, thresholds, c, subset)
                if cost is not None:
                    # the total, a label's threshold also changes which tag other tracks
                    # get (e.g. a common parent when two labels are over their thresholds)
                    key = (-(right.sum() - cost * wrong.sum()), -right.sum())
                else:
                    short = right[l] < label_targets[l]
                    key = (short, -right[l] if short else wrong[l], -right[l])
                if best is None or key < best[0]:
                    best = (key, cand)
            thresholds[l] = best[1]
    return thresholds


# per label thresholds alone, then also raised by missing frames on top of them
# (probs, c, cost), cost None keeps label_keep of the rights, else right - cost x wrong
tune_setups = {"none": (base, 0.0, None)}
for cost in args.wrong_cost:
    tune_setups[f"none, wrong cost {cost:g}"] = (base, 0.0, cost)
for c in (0.25, 0.5):
    tune_setups[f"none, missing^1 c={c}"] = (base, c, None)
for form in CAL_FORMS:
    tune_setups[form] = (cal_probs[form], 0.0, None)
for c in (0.25, 0.5):
    tune_setups[f"global T, missing^1 c={c}"] = (cal_probs["global T"], c, None)
tuned = {
    name: fitted(
        "thresholds", name, lambda: tune_thresholds(probs, c, short_fit, cost=cost)
    )
    for name, (probs, c, cost) in tune_setups.items()
}
_, cr_now, cw_now, _ = summarise(base, short_tracks, scaled=True)
rows = [["current (scaled)", cr_now, cw_now, ""]]
for name, thresholds in tuned.items():
    probs, c, _ = tune_setups[name]
    right, wrong = label_counts(probs, thresholds, c)
    rows.append(
        [
            name,
            int(right.sum()),
            int(wrong.sum()),
            round(100 * (wrong.sum() / cw_now - 1), 1),
        ]
    )
table(
    "per_label_tuned",
    f"short tracks with per label thresholds tuned to keep {args.label_keep:.0%} of each "
    f"label's current confident right, or for wrong cost setups to maximise right - "
    f"cost x wrong, {args.score} score ({cv_name}):",
    ["Method", "Confident right", "Confident wrong", "Wrong change % vs current"],
    rows,
)
table(
    "tuned_thresholds",
    f"tuned short track thresholds per label, {args.score} score:",
    ["Label", "Current"] + list(tuned),
    [
        [labels[l], round(PLAIN[l], 2)] + [round(t[l], 2) for t in tuned.values()]
        for l in range(len(labels))
    ],
)

OFFSETS = np.linspace(-0.3, 0.5, 81)


def tune_offset(probs, subset, cost=None):
    # one amount added to every label's threshold, keeping label_keep of the total right,
    # or with cost maximising total right - cost x total wrong
    target = targets_for(subset).sum()
    best = None
    for offset in OFFSETS:
        right, wrong = label_counts(
            probs, np.clip(PLAIN + offset, 0, 1.01), 0.0, subset
        )
        if cost is not None:
            key = (-(right.sum() - cost * wrong.sum()), -right.sum())
        else:
            short = right.sum() < target
            key = (short, -right.sum() if short else wrong.sum(), -right.sum())
        if best is None or key < best[0]:
            best = (key, offset)
    return np.clip(PLAIN + best[1], 0, 1.01)


# tune on the other folds and score the held out devices, so the thresholds are judged
# on cameras and stations they were not tuned on
# (probs, c, cost), c None is the shared offset
cv_setups = {
    "per label, none": (base, 0.0, None),
    "per label, none, missing^1 c=0.5": (base, 0.5, None),
    "per label, global T": (cal_probs["global T"], 0.0, None),
    "shared offset, none": (base, None, None),
}
in_sample = {
    "per label, none": tuned["none"],
    "per label, none, missing^1 c=0.5": tuned["none, missing^1 c=0.5"],
    "per label, global T": tuned["global T"],
    "shared offset, none": fitted(
        "thresholds", "shared offset", lambda: tune_offset(base, short_fit)
    ),
}
for cost in args.wrong_cost:
    cv_setups[f"per label, none, wrong cost {cost:g}"] = (base, 0.0, cost)
    in_sample[f"per label, none, wrong cost {cost:g}"] = tuned[f"none, wrong cost {cost:g}"]
short_arr = np.flatnonzero(SHORT)
held = {name: np.zeros(2, dtype=int) for name in cv_setups}
fold_thresholds = {name: [] for name in cv_setups}
if args.tune_dir or LOADED is not None:
    # one set of thresholds from the tune dir (or loaded), scored on run_dir
    for name, (probs, c, _) in cv_setups.items():
        right, wrong = label_counts(probs, in_sample[name], c or 0.0, short_tracks)
        held[name] += [right.sum(), wrong.sum()]
        fold_thresholds[name].append(in_sample[name])
else:
    for k in EVAL_FOLDS:
        train, test = short_arr[fold[short_arr] != k], short_arr[fold[short_arr] == k]
        if len(train) == 0 or len(test) == 0:
            continue
        for name, (probs, c, cost) in cv_setups.items():
            if c is None:
                thresholds, c = tune_offset(probs, train, cost), 0.0
            else:
                thresholds = tune_thresholds(probs, c, train, cost=cost)
            right, wrong = label_counts(probs, thresholds, c, test)
            held[name] += [right.sum(), wrong.sum()]
            fold_thresholds[name].append(thresholds)
        print(f"tuned fold {k + 1}/{len(EVAL_FOLDS)}")

NO_TUNE_SET = LOADED is not None and not args.tune_dir


def tune_set(right, wrong):
    # with --load-fit and no --tune-dir there is no tune set to score
    return ["", ""] if NO_TUNE_SET else [int(right), int(wrong)]


_, cr_fit, cw_fit, _ = summarise(base, short_fit, scaled=True)
rows = [["current (scaled)", cr_now, cw_now, "", ""] + tune_set(cr_fit, cw_fit)]
for name, (probs, c, _) in cv_setups.items():
    # scored on the tracks they were tuned on, the tune dir or in sample
    right, wrong = label_counts(probs, in_sample[name], c or 0.0, short_fit)
    rows.append(
        [name, int(held[name][0]), int(held[name][1])]
        + [round(100 * (held[name][0] / cr_now - 1), 1)]
        + [round(100 * (held[name][1] / cw_now - 1), 1)]
        + tune_set(right.sum(), wrong.sum())
    )
table(
    "tuned_cv",
    f"short tracks, thresholds tuned on {'the tune dir' if args.tune_dir else 'the other folds'} "
    f"and scored on held out tracks ({cv_name}, keep {args.label_keep:.0%}, "
    f"{args.score} score):",
    ["Method", "Held out confident right", "Held out confident wrong"]
    + ["Right change % vs current", "Wrong change % vs current"]
    + ["Tune set right", "Tune set wrong"],
    rows,
)
spread = np.array(fold_thresholds["per label, none"])
table(
    "tuned_thresholds_cv",
    "per label short track thresholds (none), tuned on the tune set and the range "
    "across folds:",
    ["Label", "Short tracks to tune on", "Current", "Tuned", "Fold min", "Fold max"],
    [
        [labels[l], "" if NO_TUNE_SET else int((Y[short_fit] == l).sum())]
        + [round(PLAIN[l], 2)]
        + [round(in_sample["per label, none"][l], 2)]
        + (
            [round(spread[:, l].min(), 2), round(spread[:, l].max(), 2)]
            if len(spread)
            else ["", ""]
        )
        for l in range(len(labels))
    ],
)


def pct_change(new, old):
    return round(100 * (new / old - 1), 1) if old else ""


# per label on the scored short tracks, right by true label, wrong by the label called
tag_now, conf_now, ok_now = decide(base, short_tracks, 1.0)
right_now = np.bincount(Y[short_tracks][conf_now & ok_now], minlength=len(labels))
wrong_now = np.bincount(tag_now[conf_now & ~ok_now], minlength=len(tag_names))
right_tuned, wrong_tuned = label_counts(base, in_sample["per label, none"])
right_c, wrong_c = label_counts(
    base, in_sample["per label, none, missing^1 c=0.5"], 0.5
)
cost_counts = {
    cost: label_counts(base, in_sample[f"per label, none, wrong cost {cost:g}"])
    for cost in args.wrong_cost
}
scored_counts = np.bincount(Y[short_tracks], minlength=len(labels))
rows = []
for l in sorted(range(len(labels)), key=lambda l: -scored_counts[l]):
    rows.append(
        [labels[l], int(scored_counts[l]), round(in_sample["per label, none"][l], 2)]
        + [int(right_now[l]), int(right_tuned[l]), pct_change(right_tuned[l], right_now[l])]
        + [int(wrong_now[l]), int(wrong_tuned[l]), pct_change(wrong_tuned[l], wrong_now[l])]
        + [int(right_c[l]), int(wrong_c[l])]
        + [int(v[l]) for counts in cost_counts.values() for v in counts]
    )
rows.append(
    ["total", int(scored_counts.sum()), ""]
    + [int(right_now.sum()), int(right_tuned.sum())]
    + [pct_change(right_tuned.sum(), right_now.sum())]
    + [int(wrong_now[: len(labels)].sum()), int(wrong_tuned.sum())]
    + [pct_change(wrong_tuned.sum(), wrong_now[: len(labels)].sum())]
    + [int(right_c.sum()), int(wrong_c.sum())]
    + [int(v.sum()) for counts in cost_counts.values() for v in counts]
)
table(
    "tuned_per_label",
    f"short tracks per label, current scaled thresholds vs per label thresholds tuned on "
    f"{'the tune dir' if args.tune_dir else 'these tracks'} ({cv_name}, "
    "right by true label, wrong by the label it was called):",
    ["Label", "Short tracks", "Tuned threshold", "Right current", "Right tuned"]
    + ["Right change %", "Wrong current", "Wrong tuned", "Wrong change %"]
    + ["Right tuned c=0.5", "Wrong tuned c=0.5"]
    + [
        f"{side} wrong cost {cost:g}"
        for cost in args.wrong_cost
        for side in ("Right", "Wrong")
    ],
    rows,
)



def trade(d_right, d_wrong):
    # how many right each wrong removed (or added) costs (or gains)
    if d_wrong < 0:
        if d_right <= 0:
            return f"{-d_right / -d_wrong:.2f} right lost per wrong removed"
        return "more right and fewer wrong"
    if d_wrong > 0:
        if d_right > 0:
            return f"{d_right / d_wrong:.2f} right gained per wrong added"
        return "fewer right and more wrong"
    return "same wrong"


# totals on the scored short tracks, as shares of all of them and the right per wrong
total = len(short_tracks)
summary_setups = {"current (scaled)": (int(right_now.sum()), int(wrong_now[: len(labels)].sum()))}
summary_setups[f"keep {args.label_keep:.0%}"] = (int(right_tuned.sum()), int(wrong_tuned.sum()))
for cost, (r, w) in cost_counts.items():
    summary_setups[f"wrong cost {cost:g}"] = (int(r.sum()), int(w.sum()))
base_right, base_wrong = summary_setups["current (scaled)"]
rows = []
for name, (r, w) in summary_setups.items():
    rows.append(
        [name, r, round(100 * r / total, 1), w, round(100 * w / total, 1)]
        + [r - base_right, w - base_wrong]
        + [trade(r - base_right, w - base_wrong) if name != "current (scaled)" else ""]
        + [round(r - cost * w) for cost in args.wrong_cost]
    )
table(
    "tuned_summary",
    f"short tracks totals ({total} tracks"
    + (f", ignoring {', '.join(args.ignore_mistakes)}" if IGNORED else "")
    + f", {cv_name}):",
    ["Setup", "Confident right", "Right % of short", "Confident wrong", "Wrong % of short"]
    + ["Right change", "Wrong change", "Trade vs current"]
    + [f"Right - {cost:g} x wrong" for cost in args.wrong_cost],
    rows,
)

if args.save_thresholds:
    # thresholds for tracks with 1 prediction and fewer than short_max_frames frames
    saved = {
        "short_max_frames": SEGMENT_FRAMES,
        "label_keep": args.label_keep,
        "tuned_on": args.tune_dir or args.run_dir,
        "short_thresholds": {
            l: round(float(v), 3) for l, v in zip(labels, in_sample["per label, none"])
        },
        "methods": {
            name: {l: round(float(v), 3) for l, v in zip(labels, thresholds)}
            for name, thresholds in in_sample.items()
        },
    }
    with open(args.save_thresholds, "w") as f:
        json.dump(saved, f, indent=4)
    print("saved", os.path.abspath(args.save_thresholds))

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
            + [
                round(t["F"][:, 0].max(), 1),
                round(t["F"][:, 2].mean()),
                t["num_frames"],
            ]
        )
table(
    "wrong_short_tracks",
    f"wrong short tracks, current vs global T with plain thresholds ({cv_name}):",
    ["File", "Track", "Truth", "Predicted before", "Conf before", "Confident before"]
    + ["Predicted after", "Conf after", "Confident after"]
    + ["Max disp", "Mean mass", "Frames"],
    rows,
)

finish()
