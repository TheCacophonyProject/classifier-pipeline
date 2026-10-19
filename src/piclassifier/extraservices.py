import logging
from multiprocessing import Process

from piclassifier import utils


# saves import in main process
def _postprocess_main(parse_incomplete, only_new_files, exit_event):
    from piclassifier.postprocess import main

    main(
        parse_incomplete=parse_incomplete,
        only_new_files=only_new_files,
        exit_event=exit_event,
    )


def run_postprocess(parse_incomplete=True, only_new_files=False, exit_event=None):
    p_processor = Process(
        target=_postprocess_main,
        args=(
            parse_incomplete,
            only_new_files,
            exit_event,
        ),
    )
    p_processor.start()
    return p_processor


def _classifier_main():
    from .servemodel import main

    main()


def run_classifier():
    p_processor = Process(
        target=_classifier_main,
        args=(),
    )
    p_processor.start()
    return p_processor


def start_service(start_fn):
    return {"process": start_fn(), "start_fn": start_fn}


def watch_services(services, stop_event):
    while not stop_event.wait(5 * 60):
        for service in services:
            process = service["process"]
            if not process.is_alive():
                logging.warning(
                    "Service %s died (exit code %s), restarting",
                    service["start_fn"].__name__,
                    process.exitcode,
                )
                try:
                    utils.kill_process_with_timeout(process, timeout=2)
                except:
                    pass
                service["process"] = service["start_fn"]()
