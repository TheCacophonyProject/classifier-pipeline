import numpy as np

THUMBNAIL_SIZE = 30


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
    best_frame = median_mass_i

    return score, best_frame


def best_trackless_region(clip):
    best_region = None
    # if we have regions take best mass
    for regions in clip.region_history:
        for region in regions:
            if best_region is None or region.mass > best_region.mass:
                best_region = region
    if best_region is not None:
        return best_region

    # take region with greatest filtered mean values, and
    # if zero take thermal mean values
    best_frame = np.argmax(clip.stats.frame_stats_mean)
    frame_height, frame_width = best_frame.shape
    best_filtered = best_frame - clip.background
    best_region = None
    for y in range(frame_height - THUMBNAIL_SIZE):
        for x in range(frame_width - THUMBNAIL_SIZE):
            thermal_sum = np.mean(
                best_frame[y : y + THUMBNAIL_SIZE, x : x + THUMBNAIL_SIZE]
            )
            filtered_sum = np.mean(
                best_filtered[y : y + THUMBNAIL_SIZE, x : x + THUMBNAIL_SIZE]
            )
            if best_region is None:
                best_region = ((y, x), filtered_sum, thermal_sum)
            elif best_region[1] > 0:
                if best_region[1] < filtered_sum:
                    best_region = ((y, x), thermal_sum, filtered_sum)
            elif best_region[2] < thermal_sum:
                best_region = ((y, x), thermal_sum, filtered_sum)
    return Region(
        best_region[0][1],
        best_region[0][1],
        THUMBNAIL_SIZE,
        THUMBNAIL_SIZE,
        frame_number=best_frame,
    )


def get_thumbnail(clip, predictions_per_model):
    segment_width = 5

    tag, predictions = visit_tag(clip, predictions_per_model)
    if tag is None:
        best_region = best_trackless_region(clip)
    else:
        best_region = best_predicted_region(clip, predictions_per_model)
    import matplotlib.pyplot as plt

    f, axarr = plt.subplots(1, 2)
    f = clip.frame_buffer.get_frame(best_region.frame_number)
    return best_region


def best_predicted_region(clip, predictions_per_model):
    predictions = list(predictions_per_model.values())[0].prediction_per_track.values()
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
    best_frame = best_score[3]
    return best_track.bounds_history[best_frame]
