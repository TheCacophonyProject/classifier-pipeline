import os
import json
from datetime import datetime, timedelta
import dateutil.parser


null_tags = ["false-positive", "none", "no-tag"]


class TrackResult:
    def __init__(self, track_record, clip_start, clip_end):
        """Creates track result from track stats entry."""
        self.start_time = clip_start + timedelta(seconds=track_record["start_s"])
        self.end_time = clip_end + timedelta(seconds=track_record["end_s"])
        self.label = track_record["label"]
        self.score = track_record["confidence"]
        self.clarity = track_record["clarity"]
        if self.label in null_tags:
            self.label = "none"

    def __repr__(self):
        return "{} {:.1f} clarity {:.1f}".format(
            self.label, self.score * 10, self.clarity * 10
        )

    @property
    def duration(self):
        return (self.end_time - self.start_time).total_seconds()

    @property
    def confidence(self):
        """The tracks 'confidence' level which is a combination of the score and clarity."""
        score_uncertainty = 1 - self.score
        clarity_uncertainty = 1 - self.clarity
        return 1 - ((score_uncertainty * clarity_uncertainty) ** 0.5)

    def print_tree(self, level=0):
        print("\t" * level + "-" + str(self))


class ClipResult:
    def __init__(self, full_path):
        """Initialise a clip result record from given stats file."""
        self.stats = read_stats_file(full_path)

        self.start_time = dateutil.parser.parse(self.stats["start_time"]) + timedelta(
            hours=13
        )
        self.end_time = dateutil.parser.parse(self.stats["end_time"]) + timedelta(
            hours=13
        )
        self.tracks = [
            TrackResult(track, self.start_time, self.end_time)
            for track in self.stats["tracks"]
        ]

        self.source = os.path.basename(full_path)

        self.camera = self.stats.get("camera", "none")
        self.true_tag = self.stats.get("original_tag", "unknown")

        if self.true_tag in null_tags:
            self.true_tag = "none"

        self.classifier_best_guess, self.classifier_best_score = self.get_best_guess()

    def get_best_guess(self):
        """Returns the best guess from classification data."""
        class_confidences = {}
        best_confidence = 0
        best_label = "none"

        for track in self.tracks:
            label = track.label

            confidence = track.confidence

            # we weight the false-positives lower as if they co-occur with an animals we want the animals to come
            # across
            if label == "none":
                confidence *= 0.5

            print(label, confidence)

            class_confidences[label] = max(
                class_confidences.get(label, 0.0), confidence
            )

            confidence = class_confidences[label]

            if confidence > best_confidence:
                best_label = label
                best_confidence = confidence

        return best_label, best_confidence

    @property
    def duration(self):
        return (self.end_time - self.start_time).total_seconds()

    def __repr__(self):
        return "{} {} {:.1f}".format(
            self.true_tag, self.classifier_best_guess, self.classifier_best_score * 10
        )

    def print_tree(self, level=0):
        print("\t" * level + "-" + str(self))
        for track in self.tracks:
            track.print_tree(level + 1)


class VisitResult:
    """Represents a visit."""

    def __init__(self, first_clip):
        """First clip is used a basis for this visit.  All other clips should have the same camera and true_tag."""
        self.clips = [first_clip]

    def add_clip(self, clip):
        """Adds a clip to the clip list.  Clips are maintained start_time sorted order."""
        self.clips.append(clip)
        self.clips.sort(key=lambda clip: clip.start_time)

    @property
    def camera(self):
        return self.clips[0].camera

    @property
    def true_tag(self):
        return self.clips[0].true_tag

    @property
    def start_time(self):
        if len(self.clips) == 0:
            return 0.0
        return self.clips[0].start_time

    @property
    def end_time(self):
        if len(self.clips) == 0:
            return 0.0
        return self.clips[-1].end_time

    @property
    def mid_time(self):
        if len(self.clips) == 0:
            return 0.0
        return self.start_time + timedelta(seconds=(self.duration / 2))

    @property
    def duration(self):
        """Duration of visit in seconds."""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def predicted_tag(self):
        """Returns the predicted tag based on best guess from individual clips."""
        sorted_clips = sorted(self.clips, key=lambda x: x.classifier_best_score)
        best_guess = sorted_clips[-1].classifier_best_guess
        return best_guess

    @property
    def predicted_confidence(self):
        """Returns the predicted tag based on best guess from individual clips."""
        sorted_clips = sorted(self.clips, key=lambda x: x.classifier_best_score)
        best_guess = sorted_clips[-1].classifier_best_score
        return best_guess

    def __repr__(self):
        return "{} {} {:.1f}".format(
            self.true_tag, self.predicted_tag, self.predicted_confidence * 10
        )

    def print_tree(self, level=0):
        print("\t" * level + "-" + str(self))
        for clip in self.clips:
            clip.print_tree(level + 1)


def read_stats_file(full_path):
    """reads in given stats file."""
    stats = json.load(open(full_path, "r"))

    return stats
