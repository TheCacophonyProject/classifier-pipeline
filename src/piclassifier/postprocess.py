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
from piclassifier import service
from functools import partial
import threading
import dbus
from gi.repository import GLib
from piclassifier.utils import startup_network_classifier, is_service_running


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


def rec_callback(dt, is_recording, set_function):
    if is_recording:
        logging.info("Recording started pausing processing")
    else:
        logging.info("Recording ended resuming processing")
    set_function(is_recording)


def service_started():
    logging.info("Service started")
    # if we receive this means we need to reconnect, signals still work
    # but calling methods wont


def main():
    init_logging()
    config = Config.load_from_file()
    thermal_config = ThermalConfig.load_from_file()
    output_dir = Path(thermal_config.recorder.output_dir)
    reprocess_dir = output_dir / "postprocess"
    reprocess_dir.mkdir(exist_ok=True)
    network_model = get_model(thermal_config, config)
    if network_model is None:
        logging.info("No network model exiting")
        sys.exit(0)

    process_queue = multiprocessing.Queue()
    dir_watcher = DirWatcher(process_queue)
    observer = Observer()
    reprocess_files = list(reprocess_dir.glob("*.cptv"))
    for cptv_f in reprocess_files:
        logging.info("Adding existing %s", cptv_f)
        process_queue.put(cptv_f)

    postprocess = thermal_config.motion.postprocess
    pending_exit = False
    if not postprocess:
        if len(reprocess_files) == 0:
            logging.info(
                "No files to post process and config is not set to post process, exiting"
            )
            return
        logging.info(
            "Postprocessing stale files then exiting as config is not set to postprocess"
        )
        pending_exit = True
    else:
        logging.info("Watching %s", reprocess_dir)
        observer.schedule(dir_watcher, reprocess_dir, recursive=False)
        observer.start()
        config.validate()
    clip_classifier = ClipClassifier(
        config,
        network_model,
        keep_original_predictions=True,
        tracking_events=thermal_config.motion.postprocess_events,
    )

    callback_fn = partial(rec_callback, set_function=clip_classifier.set_is_recording)
    bus = None
    dbus_object = None
    need_dbus = thermal_config.motion.postprocess_events

    if need_dbus:
        max_attempts = 3
        attempt = 1
        while bus is None:
            try:
                dbus_object, bus, thread = connect_to_dbus(callback_fn)
            except Exception as ex:
                logging.info(
                    "Couldn't connecto dbus waiting 20 seconds and trying again",
                    exc_info=True,
                )
                if attempt >= max_attempts:
                    raise ex
                time.sleep(20)
            attempt += 1
    try:
        while True:
            try:
                new_file = process_queue.get(block=False)
            except:
                if pending_exit:
                    logging.info("Finished processing exit")
                    break
                time.sleep(5)
                continue
            new_file = Path(new_file)
            if not new_file.exists():
                continue
            # reprocess file

            if not is_service_running("thermal-classifier"):
                logging.info("Network classifier is not running starting it up")
                success = startup_network_classifier(True)
                if not success:
                    raise Exception("Could not start up netowrk classifier")
                # give it some time to start up
                time.sleep(5)
            try:
                if clip_classifier._is_recording:
                    while clip_classifier._is_recording:
                        logging.info(
                            "Waiting for current recording to finish before processing"
                        )
                        time.sleep(10)
                if need_dbus:
                    # ensures dbus service is always, can be invalidated if thermal-recorder restarts
                    try:
                        dbus_object = bus.get_object(
                            service.DBUS_NAME, service.DBUS_PATH
                        )
                    except:
                        logging.error(
                            "Could not connect to dbus service %s",
                            service.DBUS_NAME,
                            exc_info=True,
                        )
                        break
                clip_classifier.post_process_file(new_file, dbus_object)

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
    loop.quit()
    if observer.is_alive:
        observer.stop()


def connect_to_dbus(rec_callback):
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()
    dbus_object = bus.get_object(service.DBUS_NAME, service.DBUS_PATH)
    loop = GLib.MainLoop()

    dbus_thread = threading.Thread(target=dbus_events, args=(loop, bus, rec_callback))
    dbus_thread.start()
    return dbus_object, bus, dbus_thread


def dbus_events(loop, dbus_object, callback_fn):
    dbus_object.add_signal_receiver(
        callback_fn,
        dbus_interface=service.DBUS_NAME,
        signal_name="Recording",
    )
    dbus_object.add_signal_receiver(
        service_started,
        dbus_interface=service.DBUS_NAME,
        signal_name="ServiceStarted",
    )
    loop.run()
