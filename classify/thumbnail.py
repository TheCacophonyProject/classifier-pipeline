import math
import numpy as np
from track.region import Region
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import namedtuple
from ml_tools import tools


def best_trackless_thumb(clip):
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


#
# def get_thumbnail(clip, predictions_per_model):
#     tag, predictions = visit_tag(clip, predictions_per_model)
#     if tag is None:
#         best_region = best_trackless_region(clip)
#     else:
#         best_region = best_predicted_region(clip, tag, predictions_per_model)
#
#     return best_region

#
# def best_predicted_region(clip, visit_tag, predictions_per_model):
#     """Get the best region based of predictions and track scores"""
#     predictions = None
#     for model_predictions in predictions_per_model.values():
#         if predictions is None:
#             # set default
#             predictions = model_predictions
#         if model_predictions.model.thumbnail_model:
#             predictions = model_predictions
#             break
#     best_score = None
#     for track in clip.tracks:
#
#         pred = predictions.prediction_for(track.get_id())
#         if pred.predicted_tag() != visit_tag:
#             continue
#         score, best_frame = track_score(
#             pred,
#             track,
#         )
#         if score is None:
#             continue
#         if best_score is None or score > best_score[0]:
#             best_score = (score, track.bounds_history[best_frame])
#
#     return best_score[1]


def get_track_thumb_stats(clip, track):
    Stat = namedtuple("Stat", "region contours median_diff")
    max_mass = 0
    max_median_diff = 0
    max_contour = 0
    stats = []
    for region in track.bounds_history:
        if region.blank or region.mass == 0:
            continue
        frame = clip.frame_buffer.get_frame(region.frame_number)
        contours, _ = cv2.findContours(
            np.uint8(region.subimage(frame.mask)),
            cv2.RETR_EXTERNAL,
            # cv2.CHAIN_APPROX_SIMPLE,
            cv2.CHAIN_APPROX_TC89_L1,
        )
        if len(contours) == 0:
            # shouldnt happen
            contour_points.append(0)
        else:
            points = len(contours[0])
            if points > max_contour:
                max_contour = points

        # get mask of all pixels that are considered an animal
        filtered_sub = region.subimage(frame.mask)
        sub_mask = filtered_sub > 0

        # get the thermal values for this mask
        thermal_sub = region.subimage(frame.thermal)
        masked_thermal = thermal_sub[sub_mask]

        # get the difference in media between this and the frame median
        t_median = np.median(frame.thermal)
        masked_median = np.median(masked_thermal)
        median_diff = masked_median - t_median

        if region.mass > max_mass:
            max_mass = region.mass
        if median_diff > max_median_diff:
            max_median_diff = median_diff
        stats.append(Stat(region, points, median_diff))
    return stats, max_mass, max_median_diff, max_contour


def get_thumbanil_info(clip, track):
    stats, max_mass, max_median_diff, max_contour = get_track_thumb_stats(clip, track)
    if len(stats) == 0:
        return None, 0
    scored_frames = sorted(
        stats,
        key=lambda s: score(s, max_mass, max_median_diff, max_contour),
        reverse=True,
    )

    best_score = score(scored_frames[0], max_mass, max_median_diff, max_contour)
    return scored_frames[0], best_score


def score(stat, max_mass, max_median_diff, max_contour):
    region = stat.region
    # mass out of 40
    mass_percent = region.mass / max_mass
    mass_percent = mass_percent * 40

    # contours out of 50
    pts = stat.contours / max_contour
    pts = pts * 50

    centroid_mid = tools.eucl_distance(region.centroid, region.mid) ** 0.5
    centroid_mid *= 2
    # this will be probably between 1-10 could be worth * 1.5

    # median diff out of 50
    diff = stat.median_diff / max_median_diff
    diff = diff * 40

    score = mass_percent + pts + diff - centroid_mid

    is_along_border = (
        region.x <= 1 or region.y <= 1 or region.bottom >= 119 or region.right >= 159
    )
    # prefer frames not on border
    if is_along_border:
        score = score - 1000
    return score


def display_track(h_data, id):
    rows = len(h_data)
    columns = 10
    fig = plt.figure(figsize=(50, 50))

    fig.suptitle(
        f"{id}  Suggested thumbnails",
        fontsize=16,
    )
    offset = 0
    for heruistic in h_data:
        for i, h_info in enumerate(heruistic):
            plt.subplot(rows, columns, i + 1 + offset)
            frame = h_info[0]
            stat = h_info[1]
            region = stat.region
            is_along_border = (
                region.x <= 1
                or region.y <= 1
                or region.bottom >= 119
                or region.right >= 159
            )
            centroid_mid = tools.eucl_distance(region.centroid, region.mid) ** 0.5
            title = f"#{region.frame_number} - centroid {round(centroid_mid)} - {is_along_border}\n"
            title += f" cts {stat.contours}"
            title += f" mass {region.mass}"
            title += f" diff {stat.median_diff}"

            remove_axes(title)

            plt.imshow(region.subimage(frame.thermal))
        offset += columns

    plt.subplots_adjust(0, 0, 0.99, 0.99, 0, 0)
    plt.show()
    plt.close(fig)
    plt.close()


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


def thumbnail_debug(clip):
    for track in clip.tracks:
        stats, max_mass, max_median_diff, max_contour = get_track_thumb_stats(
            clip, track
        )

        mass_sorted = sorted(
            stats,
            key=lambda s: s.region.mass,
            reverse=True,
        )
        mass_frames = []
        for s in mass_sorted[:10]:
            frame = clip.frame_buffer.get_frame(s.region.frame_number)
            mass_frames.append((frame, s))

        contour_frames = []
        contour_sorted = sorted(
            stats,
            key=lambda s: s.contours,
            reverse=True,
        )
        for s in contour_sorted[:10]:
            frame = clip.frame_buffer.get_frame(s.region.frame_number)
            contour_frames.append((frame, s))

        median_diff = []
        median_sorted = sorted(
            stats,
            key=lambda s: s.median_diff,
            reverse=True,
        )
        for s in median_sorted[:10]:
            frame = clip.frame_buffer.get_frame(s.region.frame_number)
            median_diff.append((frame, s))

        score_sorted = sorted(
            stats,
            key=lambda s: score(s, max_mass, max_median_diff, max_contour),
            reverse=True,
        )
        scored_frames = []

        for s in score_sorted[:10]:
            frame = clip.frame_buffer.get_frame(s.region.frame_number)
            scored_frames.append((frame, s))

        h_info = [mass_frames, contour_frames, median_diff, scored_frames]
        display_track(
            h_info,
            f"{clip.get_id()}-{track.start_frame}-{track.end_frame}",
        )


#
# def visit_tag(clip, predictions_per_model):
#     """From all tracks get that tag that occurs the most, choosing any tag of an
#     animal over infinite false-positives"""
#     animal_count = {}
#     if len(clip.tracks) == 0:
#         return None, None
#     for track in clip.tracks:
#         model = list(predictions_per_model.values())[0]
#         prediction = model.prediction_for(track.get_id())
#         label = prediction.predicted_tag()
#
#         animal_count.setdefault(label, [])
#         animal_count[label].append(prediction)
#     highest_count = sorted(
#         animal_count.keys(),
#         key=lambda label: 0 if label == "false-positive" else len(animal_count[label]),
#         reverse=True,
#     )
#     return highest_count[0], animal_count[highest_count[0]]
#
#
# def track_score(pred, track):
#     """Give track a thumbnail score based of prediction std deviation of mass and mass, return the score and best frame"""
#     mass_history = [int(bound.mass) for bound in track.bounds_history]
#     segment_mass = []
#     sorted_mass = np.argsort(mass_history)
#     upperq_i = sorted_mass[int(len(sorted_mass) * 3 / 4)]
#     max_mass_i = sorted_mass[-1]
#
#     pred_confidence = pred.max_score
#     # give up to 10 points for good prediction confidence
#     pred_score = pred_confidence * 10
#
#     max_mass = mass_history[max_mass_i]
#     upperq_mass = mass_history[upperq_i]
#     # subtract points for percentage devation
#     deviation_score = -5 * np.std(mass_history) / max_mass
#     # 0 - 5 based of size
#     mass_score = min(5, upperq_mass / 16)
#     score = pred_score + deviation_score + mass_score
#     best_frame = upperq_i
#
#     return score, best_frame
#
#
