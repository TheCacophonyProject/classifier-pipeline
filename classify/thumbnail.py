import math
import numpy as np
from track.region import Region
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt


def display_track(mass_frames, contour_frames, id):
    rows = 2
    columns = 10
    fig = plt.figure(figsize=(50, 50))

    fig.suptitle(
        f"{id}  Suggested thumbnails",
        fontsize=16,
    )

    for i, frame in enumerate(mass_frames):

        plt.subplot(rows, columns, i + 1)
        remove_axes(f"#{frame[2].frame_number} - mass {frame[1]} - pts {frame[0]}")
        plt.imshow(frame[3].subimage(frame[2].thermal))

    for i, frame in enumerate(contour_frames):

        plt.subplot(rows, columns, 10 + i + 1)
        remove_axes(f"#{frame[2].frame_number} -mass {frame[1]} - pts {frame[0]}")
        plt.imshow(frame[3].subimage(frame[2].thermal))

    plt.subplots_adjust(0, 0, 0.99, 0.99, 0, 0)
    plt.show()
    plt.clf()


def remove_axes(title):

    ax = plt.gca()
    # hide x-axis
    ax.get_xaxis().set_visible(False)
    # ax.title.set_visible(False)
    ax.set_xticklabels(())
    plt.subplots_adjust(hspace=0.001)
    ax.margins(y=0)
    # hide y-axis
    ax.get_yaxis().set_visible(False)
    ax.set_title(title)


def thumbnail_for_track(clip):
    for track in clip.tracks:
        contour_points = []
        for region in track.bounds_history:
            frame = clip.frame_buffer.get_frame(region.frame_number)
            contours, heirechy = cv2.findContours(
                np.uint8(region.subimage(frame.mask)),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_TC89_KCOS,
            )
            if len(contours) == 0:
                contour_points.append(0)
                continue
            points = len(contours[0])
            contour_points.append(points)

        mass_history = [int(bound.mass) for bound in track.bounds_history]
        segment_mass = []
        sorted_mass = np.argsort(mass_history)
        upperq_i = sorted_mass[int(len(sorted_mass) * 3 / 4)]
        mass_frames = []

        num_frames = len(mass_history) - 1 - upperq_i
        gap = int(math.ceil(num_frames / 5))
        index = int(len(sorted_mass) * 3 / 4)
        for i in sorted_mass:
            print(
                "mass in sorted is",
                mass_history[i],
                track.bounds_history[i].frame_number,
            )
        for i in range(5):
            if index >= len(sorted_mass):
                index = len(sorted_mass) - 1
                # break
            actual_index = sorted_mass[index]
            print("getting index", index, len(sorted_mass))
            region = track.bounds_history[actual_index]
            frame = clip.frame_buffer.get_frame(
                track.bounds_history[actual_index].frame_number
            )
            mass_frames.append(
                (
                    contour_points[actual_index],
                    mass_history[actual_index],
                    frame,
                    region,
                )
            )
            if gap == 0:
                break
            index += gap
        contour_frames = []
        sorted_contours = np.argsort(contour_points)[::-1]
        for index in sorted_contours[:10]:
            region = track.bounds_history[index]
            frame = clip.frame_buffer.get_frame(region.frame_number)
            contour_frames.append(
                (contour_points[index], mass_history[index], frame, region)
            )
        print(
            "have",
            len(mass_frames),
            len(contour_frames),
            track.start_frame,
            track.end_frame,
        )
        display_track(
            mass_frames,
            contour_frames,
            f"{clip.get_id()}-{track.start_frame}-{track.end_frame}",
        )


def visit_tag(clip, predictions_per_model):
    """From all tracks get that tag that occurs the most, choosing any tag of an
    animal over infinite false-positives"""
    animal_count = {}
    if len(clip.tracks) == 0:
        return None, None
    for track in clip.tracks:
        model = list(predictions_per_model.values())[0]
        prediction = model.prediction_for(track.get_id())
        label = prediction.predicted_tag()

        animal_count.setdefault(label, [])
        animal_count[label].append(prediction)
    highest_count = sorted(
        animal_count.keys(),
        key=lambda label: 0 if label == "false-positive" else len(animal_count[label]),
        reverse=True,
    )
    return highest_count[0], animal_count[highest_count[0]]


def track_score(pred, track):
    """Give track a thumbnail score based of prediction std deviation of mass and mass, return the score and best frame"""
    mass_history = [int(bound.mass) for bound in track.bounds_history]
    segment_mass = []
    sorted_mass = np.argsort(mass_history)
    upperq_i = sorted_mass[int(len(sorted_mass) * 3 / 4)]
    max_mass_i = sorted_mass[-1]

    pred_confidence = pred.max_score
    # give up to 10 points for good prediction confidence
    pred_score = pred_confidence * 10

    max_mass = mass_history[max_mass_i]
    upperq_mass = mass_history[upperq_i]
    # subtract points for percentage devation
    deviation_score = -5 * np.std(mass_history) / max_mass
    # 0 - 5 based of size
    mass_score = min(5, upperq_mass / 16)
    score = pred_score + deviation_score + mass_score
    best_frame = upperq_i

    return score, best_frame


def best_trackless_region(clip):
    """Choose a frame for clips without any track"""
    best_region = None
    THUMBNAIL_SIZE = 64

    # if we have regions take best mass of un tracked regions
    for regions in clip.region_history:
        for region in regions:
            if best_region is None or region.mass > best_region.mass:
                best_region = region
    if best_region is not None:
        return best_region

    # take region with greatest filtered mean values, and
    # if zero take thermal mean values
    best_frame_i = np.argmax(clip.stats.frame_stats_mean)
    best_frame = clip.frame_buffer.get_frame(best_frame_i).thermal
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
        frame_number=best_frame_i,
    )


def get_thumbnail(clip, predictions_per_model):
    tag, predictions = visit_tag(clip, predictions_per_model)
    if tag is None:
        best_region = best_trackless_region(clip)
    else:
        best_region = best_predicted_region(clip, tag, predictions_per_model)

    return best_region


def best_predicted_region(clip, visit_tag, predictions_per_model):
    """Get the best region based of predictions and track scores"""
    predictions = None
    for model_predictions in predictions_per_model.values():
        if predictions is None:
            # set default
            predictions = model_predictions
        if model_predictions.model.thumbnail_model:
            predictions = model_predictions
            break
    best_score = None
    for track in clip.tracks:

        pred = predictions.prediction_for(track.get_id())
        if pred.predicted_tag() != visit_tag:
            continue
        score, best_frame = track_score(
            pred,
            track,
        )
        if score is None:
            continue
        if best_score is None or score > best_score[0]:
            best_score = (score, track.bounds_history[best_frame])

    return best_score[1]
