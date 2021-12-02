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
# #
# def main():
#
#     recording()
#     recording()


def recording():
    bus = dbus.SystemBus()
    try:
        beacon_dbus = bus.get_object(NAME, PATH)
        beacon_dbus.Recording()
    except:
        logging.error("Dbus beacon error ", exc_info=True)


def classification(predictions):
    byte_predictions = {}

    for track_prediction in predictions:
        logging.debug(
            "%s prediction %s",
            track_prediction.track_id,
            track_prediction.predicted_tag(),
        )
        if track_prediction.class_best_score is None:
            return
        class_best_score = (
            100
            * track_prediction.class_best_score
            / np.sum(track_prediction.class_best_score)
        )
        for i, confidence in enumerate(class_best_score):
            if dbus.Byte(i) in byte_predictions:
                byte_predictions[dbus.Byte(i)] = max(
                    dbus.Byte(int(confidence)), byte_predictions[dbus.Byte(i)]
                )
            else:
                byte_predictions[dbus.Byte(i)] = dbus.Byte(int(confidence))

    logging.debug("Sending classifcations %s ", byte_predictions)
    bus = dbus.SystemBus()
    try:
        beacon_dbus = bus.get_object(NAME, PATH)
        beacon_dbus.Classification(byte_predictions)
    except:
        logging.error("Dbus beacon error ", exc_info=True)


#
# if __name__ == "__main__":
#     main()
