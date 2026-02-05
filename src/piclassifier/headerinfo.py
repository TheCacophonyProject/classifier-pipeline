"""
HeaderInfo describes a thermal cameras specs.
When a thermal camera first connects to the socket it will send some header
information describing it's specs e.g. Resolution, Frame rate
"""

import yaml
import attr


@attr.s
class HeaderInfo:
    X_RESOLUTION = "ResX"
    Y_RESOLUTION = "ResY"
    FPS = "FPS"
    MODEL = "Model"
    BRAND = "Brand"
    PIXEL_BITS = "PixelBits"
    FRAME_SIZE = "FrameSize"
    SERIAL = "CameraSerial"
    FIRMWARE = "Firmware"
    medium_power = attr.ib(default = False)
    res_x = attr.ib(default=160)
    res_y = attr.ib(default = 120)
    fps = attr.ib(default = 9)
    brand = attr.ib(default = "lepton")
    model = attr.ib(default="3.5")
    frame_size = attr.ib(default = 39040)
    pixel_bits = attr.ib(default = 16)
    serial = attr.ib(default = "12")
    firmware = attr.ib(default = "12")

    @classmethod
    def parse_header(cls, raw_string):
        if raw_string =="medium":
            return cls(medium_power=True)
        raw = yaml.safe_load(raw_string)
        
        headers = cls(
            res_x=raw.get(HeaderInfo.X_RESOLUTION),
            res_y=raw.get(HeaderInfo.Y_RESOLUTION),
            fps=raw.get(HeaderInfo.FPS),
            brand=raw.get(HeaderInfo.BRAND),
            model=raw.get(HeaderInfo.MODEL),
            serial=raw.get(HeaderInfo.SERIAL),
            frame_size=raw.get(HeaderInfo.FRAME_SIZE),
            pixel_bits=raw.get(HeaderInfo.PIXEL_BITS),
            firmware=raw.get(HeaderInfo.FIRMWARE),
        )
        headers.firmware = str(headers.firmware)
        if headers.res_x and headers.res_y:
            if not headers.pixel_bits and headers.frame_size:
                headers.pixel_bits = int(
                    8 * headers.frame_size / (headers.res_x * headers.res_y)
                )
            elif not headers.frame_size and headers.pixel_bits:
                headers.frame_size = int(
                    headers.res_x * headers.res_y * headers.pixel_bits / 8
                )
        headers.validate()

        return headers

    def validate(self):
        if not (self.res_x and self.res_y and self.fps and self.pixel_bits):
            raise ValueError(
                "header info is missing a required field ({}, {}, {} and/or {})".format(
                    HeaderInfo.X_RESOLUTION,
                    HeaderInfo.Y_RESOLUTION,
                    HeaderInfo.FPS,
                    HeaderInfo.PIXEL_BITS,
                )
            )
        return True

    def as_dict(self):
        return attr.asdict(self)
