import argparse

from cptv import CPTVReader

from config.config import Config
from ml_tools.logs import init_logging
from piclassifier.motiondetector import MotionDetector
from config.thermalconfig import ThermalConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("cptv", help="a CPTV file to detect motion")
    parser.add_argument("-c", "--config-file", help="Path to config file to use")
    parser.add_argument("--thermal-config-file", help="Path to pi-config file (config.toml) to use")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    init_logging()

    config = Config.load_from_file(args.config_file)
    thermal_config = ThermalConfig.load_from_file(args.thermal_config_file)
    res_x = config.res_x
    res_y = config.res_y
    print("detecting on  " + args.cptv)
    motion_detector = MotionDetector(
        res_x,
        res_y,
        thermal_config,
        config.tracking.dynamic_thresh,
        None,
    )
    with open(args.cptv, "rb") as f:
        reader = CPTVReader(f)
        for i, frame in enumerate(reader):
            motion_detector.process_frame(frame)


if __name__ == "__main__":
    main()
