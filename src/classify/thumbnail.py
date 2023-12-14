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
                best_region = ((x, y), filtered_sum, thermal_sum)
            elif best_region[1] > 0:
                if best_region[1] < filtered_sum:
                    best_region = ((x, y), thermal_sum, filtered_sum)
            elif best_region[2] < thermal_sum:
                best_region = ((x, y), thermal_sum, filtered_sum)
    centroid = (
        best_region[0][0] + THUMBNAIL_SIZE // 2,
        best_region[0][1] + THUMBNAIL_SIZE // 2,
    )
    return Region(
        best_region[0][0],
        best_region[0][1],
        THUMBNAIL_SIZE,
        THUMBNAIL_SIZE,
        frame_number=best_frame_i,
        centroid=centroid,
    )


Stat = namedtuple("Stat", "region contours median_diff")


def get_track_thumb_stats(clip, track):
    max_mass = 0
    max_median_diff = 0
    min_median_diff = 0
    max_contour = 0
    stats = []
    for region in track.bounds_history:
        if region.blank or region.mass == 0:
            continue
        frame = clip.frame_buffer.get_frame(region.frame_number)
        if frame is None:
            continue
        contour_image = frame.filtered if frame.mask is None else frame.mask
        contours, _ = cv2.findContours(
            np.uint8(region.subimage(contour_image)),
            cv2.RETR_EXTERNAL,
            # cv2.CHAIN_APPROX_SIMPLE,
            cv2.CHAIN_APPROX_TC89_L1,
        )
        if len(contours) == 0:
            # shouldnt happen
            # contour_points.append(0)
            continue
        else:
            points = len(contours[0])
            if points > max_contour:
                max_contour = points

        # get mask of all pixels that are considered an animal
        filtered_sub = region.subimage(contour_image)
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
        if median_diff < min_median_diff:
            min_median_diff = median_diff
        stats.append(Stat(region, points, median_diff))
    return stats, max_mass, max_median_diff, min_median_diff, max_contour


def get_thumbnail_info(clip, track):
    (
        stats,
        max_mass,
        max_median_diff,
        min_median_diff,
        max_contour,
    ) = get_track_thumb_stats(clip, track)
    if len(stats) == 0:
        if len(track.bounds_history) == 0:
            return None, 0
        return Stat(track.bounds_history[0], 0, 0), 0
    scored_frames = sorted(
        stats,
        key=lambda s: score(s, max_mass, max_median_diff, min_median_diff, max_contour),
        reverse=True,
    )

    best_score = score(
        scored_frames[0], max_mass, max_median_diff, min_median_diff, max_contour
    )
    return scored_frames[0], best_score


def score(stat, max_mass, max_median_diff, min_median_diff, max_contour):
    region = stat.region
    # mass out of 40
    mass_percent = region.mass / max_mass
    mass_percent = mass_percent * 40

    # contours out of 50
    pts = stat.contours / max_contour
    pts = pts * 50

    centroid_mid = tools.eucl_distance_sq(region.centroid, region.mid) ** 0.5
    centroid_mid *= 2
    # this will be probably between 1-10 could be worth * 1.5
    # median diff out of 50
    if max_median_diff == 0:
        # not sure to zero or what here
        diff = 0
        if min_median_diff != 0:
            diff = stat.median_diff + abs(min_median_diff)
            diff = diff / abs(min_median_diff)
            diff = diff * 40
    else:
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


# just for testing
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
            centroid_mid = tools.eucl_distance_sq(region.centroid, region.mid) ** 0.5
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
        (
            stats,
            max_mass,
            max_median_diff,
            min_median_diff,
            max_contour,
        ) = get_track_thumb_stats(clip, track)

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
            key=lambda s: score(
                s, max_mass, max_median_diff, min_median_diff, max_contour
            ),
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
