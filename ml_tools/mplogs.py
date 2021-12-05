# You'll need these imports in your own code
import logging
import logging.handlers
import multiprocessing
from multiprocess import Queue, Process

# Next two import lines for this demo only
from random import choice, random
import time

#
# Because you'll want to define the logging configurations for listener and workers, the
# listener and worker process functions take a configurer parameter which is a callable
# for configuring logging for that process. These functions are also passed the queue,
# which they use for communication.
#
# In practice, you can configure the listener however you want, but note that in this
# simple example, the listener does not apply level or filter logic to received records.
# In practice, you would probably want to do this logic in the worker processes, to avoid
# sending events which would be filtered out between processes.
#
# The size of the rotated files is made small so you can see the results easily.
def listener_configurer():
    root = logging.getLogger()
    h = logging.handlers.RotatingFileHandler("train.log", "a", 10000 * 100000, 0)
    f = logging.Formatter(
        "%(asctime)s %(process)-10s %(name)s %(levelname)-8s %(message)s"
    )
    h.setFormatter(f)
    root.addHandler(h)


# This is the listener process top-level loop: wait for logging events
# (LogRecords)on the queue and handle them, quit when you get a None for a
# LogRecord.
def listener_process(queue, configurer):
    configurer()
    while True:
        try:
            record = queue.get()
            if (
                record is None
            ):  # We send this as a sentinel to tell the listener to quit.
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:
            import sys, traceback

            print("Whoops! Problem:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


# The worker configuration is done at the start of the worker process run.
# Note that on Windows you can't rely on fork semantics, so each process
# will run the logging configuration code when it starts.
def worker_configurer(queue):
    h = logging.handlers.QueueHandler(queue)  # Just the one handler needed
    root = logging.getLogger()
    root.addHandler(h)
    # send all messages, for demo; no other level or filter logic applied.
    root.setLevel(logging.DEBUG)


# This is the worker process top-level loop, which just logs ten events with
# random intervening delays before terminating.
# The print messages are just so you know it's doing something!
def worker_process(queue, configurer):
    configurer(queue)
    name = multiprocessing.current_process().name
    print("Worker started: %s" % name)
    for i in range(10):
        time.sleep(random())
        logger = logging.getLogger()
        level = choice(LEVELS)
        message = choice(MESSAGES)
        logger.log(level, message)
    print("Worker finished: %s" % name)


def init_logging():
    queue = Queue(-1)
    listener = Process(target=listener_process, args=(queue, listener_configurer))
    listener.start()
    return queue, listener


# # Here's where the demo gets orchestrated. Create the queue, create and start
# # the listener, create ten workers and start them, wait for them to finish,
# # then send a None to the queue to tell the listener to finish.
# def main():
#     queue = multiprocessing.Queue(-1)
#     listener = multiprocessing.Process(target=listener_process,
#                                        args=(queue, listener_configurer))
#     listener.start()
#     workers = []
#     for i in range(10):
#         worker = multiprocessing.Process(target=worker_process,
#                                          args=(queue, worker_configurer))
#         workers.append(worker)
#         worker.start()
#     for w in workers:
#         w.join()
#     queue.put_nowait(None)
#     listener.join()
#
# if __name__ == '__main__':
#     main()
