import os
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import itertools
import argparse
import seaborn as sns

from config.config import Config
from ml_tools.logs import init_logging
from ml_tools import tools
from .evaluateresults import *


def is_stats_file(filename):
    """ returns if filename is a valid stats file. """
    # note, we also have track stats files which have 4 parts, date-time-camera-track
    ext = os.path.splitext(filename)[-1].lower()
    parts = filename.split("-")
    return ext == ".txt" and len(parts) == 3


def show_confusion_matrix(
    true_class,
    pred_class,
    labels,
    normalize=True,
    title="Classification Confusion Matrix",
):

    cm = metrics.confusion_matrix(true_class, pred_class, labels=labels)

    # normalise matrix
    if normalize:
        print(cm.sum(axis=1)[np.newaxis, :])
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.matshow(cm, cmap=plt.cm.Blues)

    plt.title(title)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    ax.set_xticklabels([""] + labels, rotation=45)
    ax.set_yticklabels([""] + labels)
    ax.xaxis.set_tick_params(labeltop="off", labelbottom="on")
    plt.show()


def show_breakdown(true_class, pred_class, title="Confusion Matrix"):

    # confusion matrix
    show_confusion_matrix(true_class, pred_class, classes, normalize=False, title=title)

    # get f1 scores
    f1_scores = metrics.f1_score(true_class, pred_class, classes, average=None)

    correct = 0
    for true, pred in zip(true_class, pred_class):
        if true == pred:
            correct += 1

    print("F1 scores:")
    for class_name, f1_score in zip(classes, f1_scores):
        print("{:<20} {:.1f}".format(class_name, f1_score * 100))

    print()

    print(
        "Correctly classified {0} / {1} = {2:.2f}%".format(
            correct, len(true_class), 100 * correct / len(true_class)
        )
    )
    print(
        "Final score: {:.1f}".format(
            100 * np.sum(f1_scores) / np.count_nonzero(f1_scores)
        )
    )


def breakdown_tracks(visits):
    """ Prints out a breakdown of per track accuracy. """

    print("-" * 60)
    print("Tracks:")
    total_duration = 0
    tracks = []
    true_class = []
    pred_class = []
    for visit in visits:
        for clip in visit.clips:
            if clip.true_tag not in classes:
                print("Warning, invalid true tag", clip.true_tag)
            for track in clip.tracks:
                tracks.append(track)
                total_duration += track.duration
                true_class.append(clip.true_tag)
                pred_class.append(track.label)
                if track.label not in classes:
                    print("Warning, invalid label", track.label)

    print()
    print("Total tracks: {} {:.1f}h".format(len(tracks), total_duration / 60 / 60))

    print("-" * 60)

    show_breakdown(true_class, pred_class, "Track Confusion Matrix")


def breakdown_clips(visits):
    """ Prints out a breakdown of per clip accuracy. """

    # display each clip
    print("-" * 60)
    print("Clips:")
    errors = 0
    correct = 0
    total_duration = 0
    i = 0
    clips = []
    for visit in visits:
        for clip in visit.clips:
            i += 1
            clips.append(clip)
            if clip.true_tag != clip.classifier_best_guess:
                print(
                    "{} {} {} {:.2f} {}".format(
                        i + 1,
                        clip.true_tag,
                        clip.classifier_best_guess,
                        clip.classifier_best_score,
                        clip.source,
                    )
                )
            if clip.true_tag == clip.classifier_best_guess:
                correct += 1
            else:
                errors += 1
            total_duration += clip.duration

    print()
    print(
        "Total footage: {} clips {:.1f}h".format(len(clips), total_duration / 60 / 60)
    )

    print("-" * 60)

    true_class = [clip.true_tag for clip in clips]
    pred_class = [clip.classifier_best_guess for clip in clips]

    show_breakdown(true_class, pred_class, "Clip Confusion Matrix")


def show_error_tree(visits):
    """ Prints a tree showing predictions at the visit, clip, and track level. """
    for i, visit in enumerate(visits):
        if visit.true_tag != visit.predicted_tag:
            visit.print_tree()


def breakdown_visits(visits):
    """ Prints out breakdown of per visit accuracy. """

    # display each visit
    print("-" * 60)
    print("Visits:")
    print("-" * 60)

    correct = 0
    for i, visit in enumerate(visits):
        if visit.true_tag == visit.predicted_tag:
            correct += 1

    # confusion matrix
    true_class = [visit.true_tag for visit in visits]
    pred_class = [visit.predicted_tag for visit in visits]

    show_breakdown(true_class, pred_class, "Visit Confusion Matrix")


def show_errors_by_score(visits):
    """ Displays errors in terms of their score level. """

    # visits by score

    errors = []
    correct = []

    for visit in visits:
        if visit.true_tag == visit.predicted_tag:
            correct.append(visit.predicted_confidence * 10)
        else:
            errors.append(visit.predicted_confidence * 10)

    bin_divisions = 2
    bins = [x / bin_divisions for x in range(10 * bin_divisions + 1)]
    plt.title("Visit Errors by Confidence")
    plt.hist(correct, bins=bins, label="correct")
    plt.hist(errors, bins=bins, label="error")
    plt.legend()
    plt.show()

    print(
        "Max confidence on misclassified visit", 0 if len(errors) == 0 else max(errors)
    )

    # clips by score

    errors = []
    correct = []

    for visit in visits:
        for clip in visit.clips:
            if clip.true_tag == clip.classifier_best_guess:
                correct.append(clip.classifier_best_score * 10)
            else:
                errors.append(clip.classifier_best_score * 10)

    plt.title("Clip Errors by Confidence")
    plt.hist(correct, bins=bins, label="correct")
    plt.hist(errors, bins=bins, label="error")
    plt.legend()
    plt.show()

    # tracks by score

    errors = []
    correct = []

    for visit in visits:
        for clip in visit.clips:
            for track in clip.tracks:
                if clip.true_tag == track.label:
                    correct.append(track.confidence * 10)
                else:
                    errors.append(track.confidence * 10)

    plt.title("Track Errors by Confidence")
    plt.hist(correct, bins=bins, label="correct")
    plt.hist(errors, bins=bins, label="error")
    plt.legend()
    plt.show()


def get_visits(path, visit_threshold):
    """ Scans a folder loading all clip statstics, and formats them into visits. """
    all_records = []
    # fetch the records
    for filename in os.listdir(path):
        if is_stats_file(filename):
            record = ClipResult(os.path.join(path, filename))
            if record.classifier_best_guess in classes:
                all_records.append(record)

    # check basic stats, such as missed objects, incorrect objects.

    cameras = set([record.camera for record in all_records])

    visits = []

    for camera in cameras:

        records = [
            record
            for record in all_records
            if record.camera == camera and record.true_tag in classes
        ]

        # group clips into visits by camera
        records.sort(key=lambda x: x.start_time)

        current_visit = None
        previous_record_end = None

        for record in records:

            if not current_visit:
                current_visit = VisitResult(record)
                visits.append(current_visit)

            gap = (
                (record.start_time - previous_record_end).total_seconds()
                if previous_record_end
                else 0.0
            )

            # start a new visit if gap is too large, or tag changes.
            if gap >= visit_threshold or (record.true_tag != current_visit.true_tag):
                current_visit = VisitResult(record)
                visits.append(current_visit)
            else:
                current_visit.add_clip(record)

            previous_record_end = record.end_time

    return visits


def show_visits_over_days(visits):

    # bin visits in days
    visit_bins = {}
    for visit in visits:
        if visit.predicted_tag == "none":
            continue
        visit_midpoint = visit.start_time + timedelta(seconds=visit.duration / 2)
        date = visit_midpoint.replace(hour=0, minute=0, second=0, microsecond=0)
        offset = (visit_midpoint - date).total_seconds() / 60 / 60
        if date not in visit_bins:
            visit_bins[date] = []
        visit_bins[date].append((offset, visit))

    bins = range(0, 24)

    for date, visit_bin in visit_bins.items():
        plt.title(
            "Classifier Visit Sightings for {}".format(date.strftime("%D %Y/%m/%d"))
        )

        xs = []

        for label in classes:
            xs.append([])
            for offset, visit in visit_bin:
                if visit.predicted_tag != label:
                    continue
                xs[-1].append(offset)

        print(xs)

        for i, x in enumerate(xs):
            plt.hist(x, bins, histtype="bar", stacked=True, label=classes[i])
            ax = plt.gca()
            ax.set_ylim([0, 10])
        plt.legend()
        plt.show()


def plot_visits(visits, true_tags=False):
    """
    Plots visits over time for each camera.
    :param visits: list of visit objects
    :param true_tags: If true true (hand labeled) tags will be used instead of predicted tags
    :return:
    """

    start_date = min([visit.start_time for visit in visits])

    visits = sorted(visits, key=lambda x: x.predicted_tag)
    cameras = sorted(list(set([visit.camera for visit in visits])))

    data_x = [visit.camera for visit in visits]
    data_y = [
        (visit.start_time - start_date).total_seconds() / (60 * 60 * 24)
        for visit in visits
    ]
    data_c = [visit.true_tag if true_tags else visit.predicted_tag for visit in visits]
    data_s = [visit.duration / 60 for visit in visits]
    plt.figure(figsize=(14, 6))
    plt.title("Strip plot for week starting {}".format(start_date.strftime("%Y/%m/%d")))

    sns.stripplot(
        y=data_x,
        x=data_y,
        order=cameras,
        hue_order=classes,
        linewidth=0.5,
        jitter=0.25,
        hue=data_c,
        dodge=True,
    )

    plt.show()


def plot_camera_visits(camera, visits):
    """
    Plots visits over time for a specific camera.
    :param visits: list of visit objects
    :param true_tags: If true true (hand labeled) tags will be used instead of predicted tags
    :return:
    """

    def time_in_seconds(time):
        return (
            time - time.replace(hour=0, minute=0, second=0, microsecond=0)
        ).total_seconds()

    def map_noon(hour):
        return hour if hour < 12 else hour - 24

    visits = [visit for visit in visits if visit.camera == camera]

    x = []
    for i, class_name in enumerate(classes):
        x.append(
            [
                map_noon(time_in_seconds(visit.mid_time) / 60 / 60)
                for visit in visits
                if visit.true_tag == class_name
            ]
        )

    plt.figure(figsize=(8, 6))
    plt.title("{} activity".format(camera))

    # sns.stripplot(x=data_x, hue_order=classes, linewidth=0.5, jitter = 0.25, hue=data_c, dodge=True)
    bin_divisions = 2
    bins = range(-12, 13, 2)

    plt.hist(x, bins=bins, histtype="bar", stacked=True, label=classes)
    plt.legend()

    plt.show()


def print_summary(visits):
    """ Outputs a summary of visits.  This does not require pre-tagged data. """

    print("Found {} visits.".format(len(visits)))

    animal_visits = {}
    for label in classes:
        animal_visits[label] = 0

    for visit in visits:
        animal_visits[visit.predicted_tag] += 1

    for class_name, visit_count in animal_visits.items():
        print("{:<10} {}".format(class_name, visit_count))

    # show_visits_over_days(visits)

    # plot_visits(visits, true_tags=True)
    plot_camera_visits("akaroa09", visits)


def print_evaluation(visits):
    """ Runs through all stats files in a folder and evaluates the performance of the classifier. """
    breakdown_tracks(visits)
    breakdown_clips(visits)
    breakdown_visits(visits)
    show_errors_by_score(visits)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", help="Path to config file to use")

    parser.add_argument(
        "-x",
        "--show-extended-evaluation",
        default=False,
        action="store_true",
        help="Evalulates results against pre-tagged ground truth.",
    )

    return parser.parse_args()


def load_config(args):
    config = Config.load_from_file(args.config_file)

    if args.show_extended_evaluation:
        config.evaluate.show_extended_evaluation = True

    return config


def main():
    init_logging()
    args = parse_args()
    conf = load_config(args)

    global null_tags, classes
    null_tags = conf.evaluate.null_tags
    classes = conf.labels

    visits = get_visits(
        conf.classify.classify_folder, conf.evaluate.new_visit_threshold
    )

    if conf.evaluate.show_extended_evaluation:
        print_evaluation(visits)
    else:
        print_summary(visits)


if __name__ == "__main__":
    main()
