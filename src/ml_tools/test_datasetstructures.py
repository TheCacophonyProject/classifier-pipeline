"""
Tests for random_sections() and get_samples_by_label_urgency().

Run from repo root:  pytest src/ml_tools/test_datasetstructures.py -v
"""

import numpy as np
import pytest
from track.region import Region
from ml_tools.datasetstructures import random_sections, get_samples_by_label_urgency

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_track(num_frames, start_frame=100, base_mass=50, gap_every=None):
    """
    Build the three arrays that random_sections() expects.

    regions / mass_history are always DENSE (one entry per consecutive frame,
    matching how TrackHeader.bounds_history is stored).  frame_indices is the
    (possibly filtered) subset of frame numbers that are valid for sampling.
    """
    all_regions = [
        Region(
            0,
            0,
            10,
            10,
            centroid=(5, 5),
            mass=base_mass + i % 5,
            frame_number=start_frame + i,
        )
        for i in range(num_frames)
    ]
    regions_arr = np.array(all_regions)
    mass_history = np.array([r.mass for r in all_regions], dtype=np.float32)

    if gap_every is None:
        frame_indices = [r.frame_number for r in all_regions]
    else:
        # simulate filtered frames: remove every gap_every-th frame from the
        # valid set while keeping regions/mass_history dense
        frame_indices = [
            r.frame_number for i, r in enumerate(all_regions) if i % gap_every != 0
        ]

    return frame_indices, regions_arr, mass_history


COMMON_KWARGS = dict(
    source_file="test.cptv",
    clip_id=1,
    track_id=1,
    camera="cam",
    location="loc",
    station_id="station",
    rec_time=None,
)


def call_random_sections(label, num_frames, start_frame=100, seed=42, gap_every=None):
    fi, regions, mass = make_track(
        num_frames, start_frame=start_frame, gap_every=gap_every
    )
    return random_sections(
        label, fi, regions, mass, start_frame, seed=seed, **COMMON_KWARGS
    )


# ---------------------------------------------------------------------------
# get_samples_by_label_urgency
# ---------------------------------------------------------------------------


class TestGetSamplesByLabelUrgency:
    def test_short_clip_gives_one_sample(self):
        # 10 s clip < 18 s threshold
        samples, num_windows, _ = get_samples_by_label_urgency(
            "possum", 90, fps=9, window_length_seconds=12
        )
        assert samples == 1
        assert num_windows == 1

    def test_normal_clip_gives_multiple_windows(self):
        # 30 s clip → multiple windows
        samples, num_windows, _ = get_samples_by_label_urgency(
            "possum", 270, fps=9, window_length_seconds=12
        )
        assert num_windows > 1
        assert samples == num_windows  # common labels: samples == windows

    def test_long_clip_capped_at_max_samples(self):
        # 120 s clip, default max_samples=5
        samples, num_windows, _ = get_samples_by_label_urgency(
            "possum", 1080, fps=9, window_length_seconds=12
        )
        assert samples <= 5

    def test_rare_label_short_clip_still_gets_samples(self):
        samples, _, _ = get_samples_by_label_urgency(
            "fox", 90, fps=9, window_length_seconds=12
        )
        assert samples >= 1

    def test_rare_label_long_clip_bounded_by_frame_count(self):
        # fox is CRITICALLY_RARE: max_resample=5, bound by ceil(total_frames / (3*25))
        total_frames = 225
        samples, _, _ = get_samples_by_label_urgency(
            "fox", total_frames, fps=9, window_length_seconds=12
        )
        max_expected = max(1, int(np.ceil(total_frames / (3 * 25))))
        assert samples <= min(5, max_expected)

    def test_moderate_rare_label_capped_at_3(self):
        # MODERATE_RARE labels (e.g. chicken) cap at max_resample=3
        total_frames = 1000
        samples, _, _ = get_samples_by_label_urgency(
            "chicken", total_frames, fps=9, window_length_seconds=12
        )
        assert samples <= 3

    def test_critically_rare_label_capped_at_5(self):
        # CRITICALLY_RARE labels (e.g. fox) cap at max_resample=5
        total_frames = 1000
        samples, _, _ = get_samples_by_label_urgency(
            "fox", total_frames, fps=9, window_length_seconds=12
        )
        assert samples <= 5

    def test_critically_rare_gets_more_samples_than_moderate_rare(self):
        # For a long track, CRITICALLY_RARE allows more samples than MODERATE_RARE
        total_frames = 500
        samples_critical, _, _ = get_samples_by_label_urgency("fox", total_frames)
        samples_moderate, _, _ = get_samples_by_label_urgency("chicken", total_frames)
        assert samples_critical >= samples_moderate

    def test_unknown_label_treated_as_common(self):
        s_common, _, _ = get_samples_by_label_urgency("possum", 270)
        s_unknown, _, _ = get_samples_by_label_urgency("unknown_animal", 270)
        assert s_common == s_unknown


# ---------------------------------------------------------------------------
# random_sections — output shape and correctness
# ---------------------------------------------------------------------------


class TestRandomSectionsOutputShape:
    def test_returns_list(self):
        segs = call_random_sections("possum", 200)
        assert isinstance(segs, list)

    def test_normal_track_returns_expected_segment_count(self):
        # 200 frames at 9 fps → ~22 s → 2 windows for common labels
        segs = call_random_sections("possum", 200)
        expected, _, _ = get_samples_by_label_urgency("possum", 199)
        assert len(segs) == expected

    def test_each_segment_has_25_frames(self):
        segs = call_random_sections("possum", 200)
        for s in segs:
            assert len(s.frame_indices) == 25

    def test_no_duplicate_frames_within_segment(self):
        segs = call_random_sections("possum", 200)
        for s in segs:
            fi = list(s.frame_indices)
            assert len(fi) == len(set(fi)), "duplicate frame numbers in segment"

    def test_all_frames_within_track_bounds(self):
        start_frame = 100
        num_frames = 200
        segs = call_random_sections("possum", num_frames, start_frame=start_frame)
        for s in segs:
            for f in s.frame_indices:
                assert start_frame <= f < start_frame + num_frames

    def test_all_frames_exist_in_original_frame_indices(self):
        start_frame = 100
        num_frames = 200
        fi, regions, mass = make_track(num_frames, start_frame=start_frame)
        segs = random_sections(
            "possum", fi, regions, mass, start_frame, seed=42, **COMMON_KWARGS
        )
        fi_set = set(fi)
        for s in segs:
            for f in s.frame_indices:
                assert f in fi_set, f"frame {f} not in original frame_indices"

    def test_deterministic_with_fixed_seed(self):
        segs_a = call_random_sections("possum", 200, seed=7)
        segs_b = call_random_sections("possum", 200, seed=7)
        for a, b in zip(segs_a, segs_b):
            assert sorted(a.frame_indices) == sorted(b.frame_indices)

    def test_different_seeds_produce_different_results(self):
        segs_a = call_random_sections("possum", 200, seed=1)
        segs_b = call_random_sections("possum", 200, seed=2)
        assert any(
            sorted(a.frame_indices) != sorted(b.frame_indices)
            for a, b in zip(segs_a, segs_b)
        )


class TestRandomSectionsWithGaps:
    def test_sparse_frame_indices_all_valid(self):
        # Every 5th frame filtered — frame_indices has gaps but regions is dense
        start_frame = 100
        num_frames = 200
        fi, regions, mass = make_track(num_frames, start_frame=start_frame, gap_every=5)
        segs = random_sections(
            "possum", fi, regions, mass, start_frame, seed=42, **COMMON_KWARGS
        )
        fi_set = set(fi)
        for s in segs:
            for f in s.frame_indices:
                assert f in fi_set, f"selected filtered frame {f}"


class TestRandomSectionsRareLabels:
    def test_rare_label_segments_start_at_different_positions(self):
        # fox (CRITICALLY_RARE) with 300 frames: ceil(299/75)=4 samples
        segs = call_random_sections("fox", 300, seed=42)
        assert len(segs) == 4
        starts = [min(s.frame_indices) for s in segs]
        assert len(set(starts)) > 1, "all segments start at identical positions"

    def test_rare_label_short_clip_gets_multiple_segments(self):
        # 100-frame fox clip is "short" (< 18 s threshold → num_windows=1) but the
        # rare-label override still requests 2 samples; stride_offset = 100//25 = 4
        # so the two windows should get independent random offsets and differ
        expected_samples, num_windows, _ = get_samples_by_label_urgency("fox", 99)
        assert num_windows == 1, "should be a short clip with only one natural window"
        assert (
            expected_samples == 2
        ), "rare-label override should push samples above num_windows"

        segs = call_random_sections("fox", 100, seed=7)
        assert len(segs) == expected_samples
        starts = [min(s.frame_indices) for s in segs]
        assert (
            len(set(starts)) > 1
        ), "rare-label short-clip windows should have different offsets"
        # all frames still within track bounds
        for s in segs:
            for f in s.frame_indices:
                assert 100 <= f < 200, f"frame {f} out of bounds"


class TestGapHandlingAndWindowFallback:
    def test_segment_with_consecutive_empty_chunks_is_discarded(self):
        # A 30-frame dead zone covers ~7 chunk-widths (chunk_size ≈ 4.3 frames).
        # The first window hits 3+ consecutive empty chunks, gets aborted, and
        # the resulting short segment (<9 frames) is discarded from the output.
        start_frame = 100
        num_frames = 300
        fi, regions, mass = make_track(num_frames, start_frame=start_frame)
        segs_clean = random_sections(
            "possum", fi, regions, mass, start_frame, seed=42, **COMMON_KWARGS
        )

        fi_gap = [f for f in fi if not (start_frame + 15 <= f < start_frame + 45)]
        segs_gap = random_sections(
            "possum", fi_gap, regions, mass, start_frame, seed=42, **COMMON_KWARGS
        )

        assert len(segs_gap) < len(
            segs_clean
        ), "dead-zone window should have been discarded"
        fi_set = set(fi_gap)
        for s in segs_gap:
            for f in s.frame_indices:
                assert f in fi_set, f"frame {f} is inside the removed dead zone"

    def test_subsequent_windows_tried_after_failed_window(self):
        # Same dead zone causes the first (lowest) window to fail; the function
        # continues to the remaining four windows, collecting exactly samples-1
        # segments rather than stopping at the first failure.
        start_frame = 100
        num_frames = 500
        fi, regions, mass = make_track(num_frames, start_frame=start_frame)
        fi_gap = [f for f in fi if not (start_frame + 15 <= f < start_frame + 45)]

        expected_total, _, _ = get_samples_by_label_urgency("possum", num_frames - 1)
        segs = random_sections(
            "possum", fi_gap, regions, mass, start_frame, seed=42, **COMMON_KWARGS
        )

        assert len(segs) == expected_total - 1
        fi_set = set(fi_gap)
        for s in segs:
            for f in s.frame_indices:
                assert f in fi_set


class TestRandomSectionsEdgeCases:
    def test_short_track_does_not_crash(self):
        # Tracks shorter than chunks (25) — chunk_size ≤ 1, must not raise
        segs = call_random_sections("possum", 15)
        assert isinstance(segs, list)

    def test_segment_mass_is_positive(self):
        segs = call_random_sections("possum", 200)
        for s in segs:
            assert s.mass > 0


# ---------------------------------------------------------------------------
# Windowing math
# ---------------------------------------------------------------------------


class TestWindowingMath:
    def test_arange_produces_correct_number_of_windows(self):
        """
        np.arange(0, num_windows * half, step=half) must yield exactly
        num_windows start positions.  The start must be 0, not num_windows.
        """
        window_frames = 108  # 12 s * 9 fps
        half = window_frames // 2  # 54
        for num_windows in [1, 2, 3, 5]:
            correct = np.arange(0, num_windows * half, step=half)
            assert (
                len(correct) == num_windows
            ), f"Expected {num_windows} window starts, got {len(correct)}"
