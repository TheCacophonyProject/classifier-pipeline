# import argparse

from cptv import CPTVReader
from config.config import Config
from ml_tools.logs import init_logging
from piclassifier.cptvmotiondetector import CPTVMotionDetector
from config.thermalconfig import ThermalConfig
from piclassifier.headerinfo import HeaderInfo

#
#
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "cptv", default="clips/possum.cptv", help="a CPTV file to detect motion"
#     )
#     parser.add_argument(
#         "-c",
#         "--config-file",
#         default="tests/config.toml",
#         help="Path to config file to use",
#     )
#     parser.add_argument(
#         "--thermal-config-file",
#         default="test-config.yaml",
#         help="Path to pi-config file (config.toml) to use",
#     )
#
#     args = parser.parse_args()
#     return args


def test_motion(
    cptv_file="tests/clips/possum.cptv",
    thermal_config_file="tests/config.toml",
    config_file="tests/test-config.yaml",
):
    from pathlib import Path

    config = Config.load_from_file(config_file)
    thermal_config = ThermalConfig.load_from_file(thermal_config_file)
    print("detecting on  " + cptv_file)

    with open(cptv_file, "rb") as f:
        reader = CPTVReader(f)

        headers = HeaderInfo(
            res_x=reader.x_resolution,
            res_y=reader.y_resolution,
            fps=9,
            brand="",
            model="",
            frame_size=reader.x_resolution * reader.y_resolution * 2,
            pixel_bits=16,
            serial="",
            firmware="",
        )

        motion_detector = CPTVMotionDetector(
            thermal_config, config.tracking["thermal"].motion.dynamic_thresh, headers
        )
        for i, frame in enumerate(reader):
            motion_detector.process_frame(frame)
