import logging
from threading import Thread
from inotify_simple import INotify, flags


def monitor_file(on_modify, filename="/etc/cacophony/config.toml"):
    # create the watch in the calling thread, only reading events happens in the
    # background. INotify() runs ldconfig via subprocess, if another thread forks
    # a Process while that runs the child inherits subprocess's internal pipe and
    # INotify() never returns, so call this before starting any processes
    logging.info("Monitoring %s", filename)
    inotify = INotify()
    watch_flags = flags.CREATE | flags.MODIFY | flags.DELETE_SELF
    inotify.add_watch(filename, watch_flags)

    def read_events():
        for event in inotify.read():
            event_reasons = [str(flag) for flag in flags.from_mask(event.mask)]
            on_modify(",".join(event_reasons))

    monitor_thread = Thread(target=read_events)
    monitor_thread.daemon = True
    monitor_thread.start()
    return monitor_thread
