import dbus
from random import randint
import logging
import sys
import numpy as np

NAME = "org.cacophony.beacon"
PATH = "/org/cacophony/beacon"

#
# def init_logging(timestamps=False):
#     """Set up logging for use by various classifier pipeline scripts.
#
#     Logs will go to stderr.
#     """
#
#     fmt = "%(process)d %(thread)s:%(levelname)7s %(message)s"
#     if timestamps:
#         fmt = "%(asctime)s " + fmt
#     logging.basicConfig(
#         stream=sys.stderr, level=logging.DEBUG, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
#     )
#
#
# def main():
#     init_logging()
#     predictions = {}
#     total = 100
#     for i in range(1, 10):
#         confidence = randint(0, total)
#         predictions[dbus.Byte(i)] = dbus.Byte(confidence)
#         # .to_bytes(1, byteorder="big")
#         total -= confidence
#     classification(predictions)
#     recording()


def recording():
    bus = dbus.SystemBus()
    try:
        beacon_dbus = bus.get_object(NAME, PATH)
        beacon_dbus.Recording()
    except:
        logging.error("Dbus beacon error ", exc_info=True)


def classification(track_prediction):
    if track_prediction.class_best_score is None:
        return
    class_best_score = track_prediction.class_best_score / np.sum(
        track_prediction.class_best_score
    )
    predictions = {}
    for i, confidence in enumerate(class_best_score):
        predictions[dbus.Byte(i)] = dbus.Byte(int(confidence))
    logging.debug("Sending classifcations ", predictions)
    bus = dbus.SystemBus()
    try:
        beacon_dbus = bus.get_object(NAME, PATH)
        beacon_dbus.Classification(predictions)
    except:
        logging.error("Dbus beacon error ", exc_info=True)


#
# if __name__ == "__main__":
#     main()
