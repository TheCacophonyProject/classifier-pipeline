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
        tag = Track.get_best_human_tag(t.get("tags"), tag_precedence, min_confidence=0.8)
        if tag:
            tags[t["id"]] = tag["what"]
    return tags


tracks = []
labels = None
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
            rng = [pos[o] for o in range(lo, hi + 1) if o in pos and not pos[o]["blank"]]
            used = [pos[o] for o in fr if o in pos and not pos[o]["blank"]]
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
            feats.append((max_disp, used_disp, mean_mass, pr.max()))
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


def threshold_for(t, label_index):
    # same as TrackPrediction: per label threshold, raised for tracks with few frames
    threshold = label_thresholds.get(labels[label_index], DEFAULT_THRESHOLD)
    if scale_thresholds:
        threshold = get_scaled_thresh(threshold, t["num_frames"])
    return threshold


def is_confident(t, p):
    return p.max() >= threshold_for(t, p.argmax())


long_tracks = [i for i, t in enumerate(tracks) if len(t["P"]) >= args.long]
print(f"{len(tracks)} labelled tracks, {sum(len(t['P']) for t in tracks)} predictions")
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
files = sorted({t["file"] for t in tracks})


def pooled(theta, t, form):
    phi = FORMS[form](t["F"])
    z = phi @ theta if phi.shape[1] else np.zeros(len(t["P"]))
    w = np.exp(z - z.max())
    return (w[:, None] * t["P"]).sum(0) / w.sum()


def loss(theta, data, form, l2=0.01):
    total = 0
    for t in data:
        p = pooled(theta, t, form)
        total -= class_w[t["y"]] * np.log(p[t["y"]] + 1e-6)
    return total / len(data) + l2 * np.sum(theta**2)


def fit(data, form):
    n = FORMS[form](data[0]["F"]).shape[1]
    if n == 0:
        return np.zeros(0)
    return minimize(loss, np.zeros(n), args=(data, form), method="L-BFGS-B").x


def leave_one_recording_out(fit_fn, predict_fn):
    # fit without each recording and predict it, so every track is scored unseen
    probs = [None] * len(tracks)
    for f in files:
        theta = fit_fn([t for t in tracks if t["file"] != f])
        for i, t in enumerate(tracks):
            if t["file"] == f:
                probs[i] = predict_fn(theta, t)
    return probs


def summarise(probs, subset):
    # accuracy, confident right, confident wrong, log loss over the subset of tracks
    right = conf_right = conf_wrong = 0
    ll = 0
    for i in subset:
        t, p = tracks[i], probs[i]
        correct = p.argmax() == t["y"]
        right += correct
        if is_confident(t, p):
            conf_right += correct
            conf_wrong += not correct
        ll -= np.log(p[t["y"]] + 1e-6)
    n = max(len(subset), 1)
    return right / n, conf_right, conf_wrong, ll / n


all_tracks = list(range(len(tracks)))
print()
print("weighting (leave one recording out, confident uses scaled per label thresholds):")
print(
    "| Weighting | Formula (fit on all) | Accuracy | Confident right | Confident wrong | Log loss "
    f"| Long accuracy | Long confident right | Long confident wrong | Long log loss |"
)
print("|---|---|---|---|---|---|---|---|---|---|")
for form in FORMS:
    probs = leave_one_recording_out(
        lambda data: fit(data, form), lambda theta, t: pooled(theta, t, form)
    )
    acc, cr, cw, ll = summarise(probs, all_tracks)
    lacc, lcr, lcw, lll = summarise(probs, long_tracks)
    theta = fit(tracks, form)
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
print(f"long tracks (current mean vs disp + mass power, leave one recording out):")
probs = leave_one_recording_out(
    lambda data: fit(data, "disp + mass power"),
    lambda theta, t: pooled(theta, t, "disp + mass power"),
)
for i in long_tracks:
    t = tracks[i]
    a = pooled(np.zeros(0), t, "mean (current)")
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
# p' = softmax(log p / T), T = exp(theta . [1, phi]); changes confidence, never the argmax
CAL_FORMS = {
    "global T": lambda F: np.ones((len(F), 1)),
    "T(disp)": lambda F: np.c_[np.ones(len(F)), np.log1p(F[:, 0])],
    "T(mass)": lambda F: np.c_[np.ones(len(F)), np.log1p(F[:, 2])],
    "T(disp, mass)": lambda F: np.c_[
        np.ones(len(F)), np.log1p(F[:, 0]), np.log1p(F[:, 2])
    ],
}


def calibrated(theta, t, form):
    phi = CAL_FORMS[form](t["F"])
    T = np.exp(phi @ theta)[:, None]
    z = np.log(t["P"] + 1e-4) / T
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    return p.mean(0)


def cal_loss(theta, data, form, l2=0.01):
    total = 0
    for t in data:
        p = calibrated(theta, t, form)
        total -= class_w[t["y"]] * np.log(p[t["y"]] + 1e-6)
    return total / len(data) + l2 * np.sum(theta[1:] ** 2)


def cal_fit(data, form):
    n = CAL_FORMS[form](data[0]["F"]).shape[1]
    return minimize(cal_loss, np.zeros(n), args=(data, form), method="L-BFGS-B").x


print()
print("calibration (leave one recording out, confident uses scaled per label thresholds):")
print("| Calibration | Formula (fit on all) | Confident right | Confident wrong | Log loss |")
print("|---|---|---|---|---|")
base = [pooled(np.zeros(0), t, "mean (current)") for t in tracks]
_, cr, cw, ll = summarise(base, all_tracks)
print(f"| none (current) | – | {cr} | {cw} | {ll:.3f} |")
cal_probs = {}
for form in CAL_FORMS:
    probs = leave_one_recording_out(
        lambda data: cal_fit(data, form), lambda theta, t: calibrated(theta, t, form)
    )
    cal_probs[form] = probs
    _, cr, cw, ll = summarise(probs, all_tracks)
    theta = cal_fit(tracks, form)
    names = ["e^{:.2f}", "(d+1)^{:.2f}" if "disp" in form else "(m+1)^{:.2f}", "(m+1)^{:.2f}"]
    formula = "T = " + " · ".join(n.format(v) for n, v in zip(names, theta))
    print(f"| {form} | {formula} | {cr} | {cw} | {ll:.3f} |")

print()
print("wrong tracks, before / after T(disp, mass) (leave one recording out):")
for i, (t, b) in enumerate(zip(tracks, base)):
    if b.argmax() != t["y"]:
        a = cal_probs["T(disp, mass)"][i]
        print(
            f"  {t['file']:50s} truth={labels[t['y']]:14s} pred={labels[b.argmax()]:14s} "
            f"thresh={threshold_for(t, b.argmax()):.2f} {b.max():.2f} -> {a.max():.2f} "
            f"disp={t['F'][:, 0].max():.1f} mass={t['F'][:, 2].mean():.0f}"
        )
