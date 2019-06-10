import logging
import multiprocessing
import os
import time
from datetime import datetime


def process_job(job):
    """ Just a wrapper to pass tupple containing (extractor, *params) to the process_file method. """
    processor = job[0]
    path = job[1]
    params = job[2]

    try:
        processor.process_file(path, **params)
    except Exception:
        logging.exception("Warning - error processing job")

    time.sleep(0.001)  # apparently gives me a chance to catch the control-c


class CPTVFileProcessor:
    """
    Base class for processing a collection of CPTV video files.
    Supports a worker pool to process multiple files at once.
    """

    def __init__(self, config, tracker_config):

        self.config = config
        self.tracker_config = tracker_config

        """
        A base class for processing large sets of CPTV video files.
        """
        self.start_date = None
        self.end_date = None

        # folder to output files to
        self.output_folder = None

        # base folder for source files
        self.source_folder = None

        # number of threads to use when processing jobs.
        self.workers_threads = config.worker_threads

        # optional initializer for worker threads
        self.worker_pool_init = None

        os.makedirs(config.classify.classify_folder, mode=0o775, exist_ok=True)

    def process_file(self, filename, **kwargs):
        """ The function to process an individual file. """
        raise Exception("Process file method must be overwritten in sub class.")

    def process_all(self, root, **kwargs):
        if root is None:
            root = self.config.source_folder

        jobs = []
        for folder_path, _, files in os.walk(root):
            for name in files:
                if os.path.splitext(name)[1] == ".cptv":
                    full_path = os.path.join(folder_path, name)
                    if self.needs_processing(full_path):
                        jobs.append((self, full_path, kwargs))

        self._process_job_list(jobs)

    def needs_processing(self, filename):
        """
        Returns True if this file needs to be processed, false otherwise.
        :param filename: the full path and filename of the cptv file in question.
        :return: returns true if file should be processed, false otherwise
        """

        # check date filters
        date_part = str(os.path.basename(filename).split("-")[0])
        date = datetime.strptime(date_part, "%Y%m%d")
        if self.start_date and date < self.start_date:
            return False
        if self.end_date and date > self.end_date:
            return False

        # look to see of the destination file already exists.
        classify_name = self.get_classify_filename(filename)
        meta_filename = classify_name + ".txt"

        # if no stats file exists we haven't processed file, so reprocess
        if self.config.reprocess:
            return True
        else:
            return not os.path.exists(meta_filename)

    def _process_job_list(self, jobs, worker_pool_args=None):
        """
        Processes a list of jobs. Supports worker threads.
        :param jobs: List of jobs to process
        :param worker_pool_args: optional arguments to worker pool initializer
        """

        if self.workers_threads == 0:
            # just process the jobs in the main thread
            for job in jobs:
                process_job(job)
        else:
            # send the jobs to a worker pool
            pool = multiprocessing.Pool(
                self.workers_threads,
                initializer=self.worker_pool_init,
                initargs=worker_pool_args,
            )
            try:
                # see https://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
                pool.map(process_job, jobs, chunksize=1)
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                logging.info("KeyboardInterrupt, terminating.")
                pool.terminate()
                exit()
            except Exception:
                logging.exception("Error processing files")
            else:
                pool.close()

    def log_message(self, message):
        """ Record message in stdout.  Will be printed if verbose is enabled. """
        # note, python has really good logging... I should probably make use of this.
        if self.tracker_config.verbose:
            logging.info(message)

    def log_warning(self, message):
        """ Record warning message in stdout."""
        # note, python has really good logging... I should probably make use of this.
        logging.warning("Warning: %s", message)


if __name__ == "__main__":
    # for some reason the fork method seems to memory leak, and unix defaults to this so we
    # stick to spawn.  Also, form might be a problem as some numpy commands have multiple threads?
    multiprocessing.set_start_method("spawn")
