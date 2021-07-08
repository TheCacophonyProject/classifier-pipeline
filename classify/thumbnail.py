import numpy as np


def visit_tag(clip, predictions_per_model):
    animal_count = {}
    for track in clip.tracks:
        model = list(predictions_per_model.values())[0]
        prediction = model.prediction_for(track.get_id())
        label = prediction.predicted_tag()

        animal_count.setdefault(label, [])
        animal_count[label].append(prediction)
    if len(animal_count) == 0:
        return None, None
    highest_count = sorted(
        animal_count.keys(),
        key=lambda label: 0 if label == "false-positive" else len(animal_count[label]),
        reverse=True,
    )
    return highest_count[0], animal_count[highest_count[0]]


def track_score(pred, track):
    mass_history = [int(bound.mass) for bound in track.bounds_history]
    segment_mass = []
    sorted_mass = np.argsort(mass_history)
    median_mass_i = sorted_mass[int(len(sorted_mass) * 3 / 4)]
    max_mass_i = sorted_mass[-1]

    pred_confidence = pred.max_score
    pred_score = pred_confidence / 0.1

    max_mass = mass_history[max_mass_i]
    median_mass = mass_history[median_mass_i]
    deviation_score = -5 * np.std(mass_history) / max_mass
    mass_score = min(5, median_mass / 16)
    score = pred_score + deviation_score + mass_score
    print(
        f"mass_score {mass_score} values confidence {pred_confidence} best mass {sorted_mass[1]} max {max_mass} and medan mass {median_mass} std dev {deviation_score}"
    )
    best_frame = median_mass_i
    print(
        "best frame for track is", score, best_frame, track.get_id(), track.start_frame
    )
    return score, best_frame


def get_thumbnail(clip, predictions_per_model):
    segment_width = 5

    tag, predictions = visit_tag(clip, predictions_per_model)
    print("best tag is", tag)
    if tag is None:
        # what to do
        return
    best_score = None
    for pred in predictions:
        track = next(track for track in clip.tracks if track.get_id() == pred.track_id)
        score, best_frame = track_score(
            pred,
            track,
        )
        if score is None:
            continue
        if best_score is None or score > best_score[0]:
            best_score = (score, pred, track, best_frame)
    best_track = best_score[2]
    print("using track", best_track)
    best_frame = best_score[3]
    import matplotlib.pyplot as plt

    f, axarr = plt.subplots(1, 2)
    f = clip.frame_buffer.get_frame(best_frame + best_track.start_frame)
    f.crop_by_region(best_track.bounds_history[best_frame], out=f)
    axarr[0].imshow(f.thermal)
    axarr[1].imshow(f.filtered)
    plt.show()
