import sys
import time
import logging
from inotify_simple import INotify, flags


def monitor_file(on_modify, filename="/etc/cacophony/config.toml"):
    inotify = INotify()
    watch_flags = flags.CREATE | flags.MODIFY | flags.DELETE_SELF
    wd = inotify.add_watch("/etc/cacophony/config.toml", watch_flags)
    for event in inotify.read():
        event_reasons = [str(flag) for flag in flags.from_mask(event.mask)]
        on_modify(",".join(event_reasons))
