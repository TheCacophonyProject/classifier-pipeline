import logging
import os
import time
import multiprocessing
import traceback

def process_job(job):
    """ Just a wrapper to pass tupple containing (extractor, *params) to the process_file method. """
    processor = job[0]
    path = job[1]
    params = job[2]

    try:
        processor.process_file(path, **params)
    except Exception as e:
        print("Warning - error processing job:",e)
        traceback.print_exc()

    time.sleep(0.001) # apparently gives me a chance to catch the control-c


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

    def needs_processing(self, filename):
        """ Checks if source file needs processing. """
        return True

    def process_folder(self, folder_path, worker_pool_args=None, **kwargs):
        """Processes all files within a folder."""

        jobs = []

        print('processing',folder_path)

        for file_name in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file_name)
            if os.path.isfile(full_path) and os.path.splitext(full_path )[1].lower() == '.cptv':
                if self.needs_processing(full_path):
                    jobs.append((self, full_path, kwargs))

        self.process_job_list(jobs, worker_pool_args)

    def process_job_list(self, jobs, worker_pool_args=None):
        """
        Processes a list of jobs. Supports worker threads.
        :param jobs: List of jobs to process
        :param worker_pool_args: optional arguments to worker pool initializer
        """

        if self.workers_threads == 0:
            # just process the jobs in the main thread
            for job in jobs: process_job(job)
        else:
            # send the jobs to a worker pool
            pool = multiprocessing.Pool(self.workers_threads, initializer=self.worker_pool_init, initargs=worker_pool_args)
            try:
                # see https://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
                pool.map(process_job, jobs, chunksize=1)
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                print("KeyboardInterrupt, terminating.")
                pool.terminate()
                exit()
            except Exception:
                logging.exception("Error processing files")
            else:
                pool.close()

    def log_message(self, message):
        """ Record message in stdout.  Will be printed if verbose is enabled. """
        # note, python has really good logging... I should probably make use of this.
        if self.tracker_config.verbose: print(message)

    def log_warning(self, message):
        """ Record warning message in stdout."""
        # note, python has really good logging... I should probably make use of this.
        print("Warning:",message)

if __name__ == '__main__':
    # for some reason the fork method seems to memory leak, and unix defaults to this so we
    # stick to spawn.  Also, form might be a problem as some numpy commands have multiple threads?
    multiprocessing.set_start_method('spawn')
