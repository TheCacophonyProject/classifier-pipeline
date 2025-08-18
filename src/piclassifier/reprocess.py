import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from config.thermalconfig import ThermalConfig
from pathlib import Path
import multiprocessing
from classify.clipclassifier import ClipClassifier
from config.config import Config
import logging
from ml_tools.logs import init_logging
import time
import sys
import shutil


class DirWatcher(FileSystemEventHandler):
    def __init__(self, process_queue):
        self.process_queue = process_queue

    def on_created(self, event):
        if not event.is_directory:
            event_file = Path(event.src_path)
            if event_file.suffix == ".cptv":
                if event_file.with_suffix(".txt").exists():
                    self.process_queue.put(event_file)
            elif event_file.suffix == ".txt":
                if event_file.with_suffix(".cptv").exists():
                    self.process_queue.put(event_file.with_suffix(".cptv"))


def get_model(thermal_config, config):

    if not thermal_config.motion.run_classifier:
        logging.info("Classifier isn't configured to run in config")
        return None

    network_model = [
        model for model in config.classify.models if model.run_over_network
    ]
    if len(network_model) == 0:
        logging.info("No network model configured in classifier.yaml")
        return None
    if len(network_model) > 1:
        logging.info("Got multiple network models using first")
    return network_model[0]


def main():
    init_logging()
    config = Config.load_from_file()
    thermal_config = ThermalConfig.load_from_file()
    output_dir = Path(thermal_config.recorder.output_dir)
    reprocess_dir = output_dir / "reprocess"
    reprocess_dir.mkdir(exist_ok=True)
    network_model = get_model(thermal_config, config)
    if network_model is None:
        logging.info("No network model exiting")
        sys.exit(0)

    process_queue = multiprocessing.Queue()
    dir_watcher = DirWatcher(process_queue)
    observer = Observer()

    for cptv_f in reprocess_dir.glob("*.cptv"):
        logging.info("Adding existing %s", cptv_f)
        process_queue.put(cptv_f)

    logging.info("Watching %s", reprocess_dir)
    observer.schedule(dir_watcher, reprocess_dir, recursive=False)
    observer.start()
    config.validate()
    clip_classifier = ClipClassifier(
        config,
        network_model,
        keep_original_predictions=True,
        tracking_events=thermal_config.motion.tracking_events,
    )
    try:
        while True:
            try:
                new_file = process_queue.get(block=False)
            except:
                time.sleep(20)
                continue
            new_file = Path(new_file)

            # reprocess file
            try:
                clip_classifier.process_file_low_mem(new_file)

            except:
                logging.error("Error reprocessing %s", new_file, exc_info=True)

            if new_file.exists():
                try:
                    meta_f = new_file.with_suffix(".txt")
                    shutil.move(new_file, output_dir / new_file.name)
                    shutil.move(meta_f, output_dir / meta_f.name)
                except:
                    logging.error(
                        "Error Moving %s to processing", new_file, exc_info=True
                    )

            logging.info("Finished processing %s", meta_f)

    except KeyboardInterrupt:
        observer.stop()


if __name__ == "__main__":
    main()
