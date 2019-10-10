import argparse

from cptv import CPTVReader

from ml_tools.config import Config
from ml_tools.logs import init_logging
from piclassifier.motiondetector import MotionDetector
from piclassifier.locationconfig import LocationConfig
from piclassifier.thermalconfig import ThermalConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("cptv", help="a CPTV file to detect motion")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    init_logging()

    config = Config.load_from_file()
    thermal_config = ThermalConfig.load_from_file()
    location_config = LocationConfig.load_from_file()
    res_x = config.classify.res_x
    res_y = config.classify.res_y
    print("detecting on  " + args.cptv)
    motion_detector = MotionDetector(
        res_x,
        res_y,
        thermal_config.motion,
        location_config,
        thermal_config.recorder,
        config.tracking.dynamic_thresh,
        None,
    )
    with open(args.cptv, "rb") as f:
        reader = CPTVReader(f)
        for i, frame in enumerate(reader):
            motion_detector.process_frame(frame)


if __name__ == "__main__":
    main()
