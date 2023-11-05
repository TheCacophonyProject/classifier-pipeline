import logging
import os

from .train import train_model


# this is a good list for a full search, but will take a long time to run (days)
FULL_SEARCH_PARAMS = {
    "batch_size": [1, 2, 4, 8, 16, 32, 64],
    "learning_rate": [1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
    "l2_reg": [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
    "label_smoothing": [0, 0.01, 0.05, 0.1, 0.2],
    "keep_prob": [0, 0.1, 0.2, 0.5, 0.8, 1.0],
    "batch_norm": [True, False],
    "lstm_units": [64, 128, 256, 512],
    "enable_flow": [True, False],
    "augmentation": [True, False],
    "thermal_threshold": [-100, -20, -10, 0, 10, 20, 100],
    # these runs will be identical in their parameters, it gives a feel for the varience during training.
    "identical": [1, 2, 3, 4, 5],
}

# this checks just the important parameters, and only around areas that are likely to work well.
# I've also excluded the default values as these do not need to be tested again.
SHORT_SEARCH_PARAMS = {
    "batch_size": [8, 32],
    "l2_reg": [1e-2, 1e-3, 1e-4],
    "label_smoothing": [0, 0.05, 0.2],
    "keep_prob": [0.1, 0.4, 0.6, 1.0],
    "batch_norm": [False],
    "lstm_units": [128, 512],
    "enable_flow": [False],
    "augmentation": [False],
    "thermal_threshold": [-100, -20, -10, 0, 20, 100],
}


def axis_search(conf):
    """
    Evaluate each hyper-parameter individually against a reference.

    The idea here is to assess each parameter individually while holding all other parameters at their default.
    For optimal results this will need to be done multiple times, each time updating the defaults to their optimal
    values.
    """
    logging.info("Performing hyper parameter search.")

    results_filename = os.path.join(conf.train.train_dir, "search-results.txt")
    logging.info(f"Writing hyper-parameter search results to {results_filename}")
    tracker = JobTracker(results_filename)

    # run the reference job with default params
    run_job(tracker, "reference", conf)

    for param_name, param_values in SHORT_SEARCH_PARAMS.items():
        for param_value in param_values:
            job_name = f"{param_name}={param_value}"
            run_job(tracker, job_name, conf, {param_name: param_value})


def run_job(tracker, job_name, conf, hyper_params=None):
    """Run a job with given hyper parameters, and log its results."""
    if tracker.is_done(job_name):
        return
    if not hyper_params:
        hyper_params = {}

    print("-" * 60)
    print("Processing", job_name)
    print("-" * 60)

    model = train_model(job_name, conf, hyper_params)
    tracker.mark_done(job_name, model.eval_score, hyper_params)


class JobTracker:
    def __init__(self, filename):
        self.filename = filename
        open(self.filename, "w").close()  # Create/truncate

    def is_done(self, job_name):
        """Returns True if this job has been processed before."""
        with open(self.filename, "r") as f:
            return any(line.split(",", 1)[0] == job_name for line in f)

    def mark_done(self, job_name, score, hyper_params):
        with open(self.filename, "a") as f:
            f.write(
                "{}, {}: {}\n".format(
                    job_name,
                    score,
                    " ".join(f"{k}={v}" for (k, v) in hyper_params.items()),
                )
            )
