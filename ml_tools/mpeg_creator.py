import locale
import os
import subprocess
import logging


class MPEGCreator:
    """
    This class allows an MPEG video to be created frame by frame.

    Usage:

        m = MPEGCreator("out.mp4")
        m.next_frame(frame0)
        m.next_frame(frame1)
        ...
        m.close()

    The output from ffmpeg is available via the `output` property.
    """

    def __init__(
        self,
        filename,
        quality=21,
        fps=9,
        codec="libx264",
        bitrate="400K",
        pix_fmt="bgr24",
    ):
        self.filename = filename
        self.quality = quality
        self._ffmpeg = None
        self._output = []
        self.fps = fps
        self.codec = codec
        self.bitrate = bitrate
        self.pix_fmt = pix_fmt

    def next_frame(self, frame, shape=None):
        if self._ffmpeg is None:
            if shape is None:
                height, width, _ = frame.shape
            else:
                height, width, _ = shape
            self._ffmpeg = self._start(width, height)
        self._ffmpeg.stdin.write(frame.tobytes())

    def close(self):
        if not self._ffmpeg:
            return
        self._ffmpeg.stdin.close()

        return_code = self._ffmpeg.wait(timeout=60)
        if return_code:
            self._collect_output()
            raise Exception(
                "ffmpeg failed with error {}. output:\n{}".format(
                    return_code, self.output
                )
            )

    @property
    def output(self):
        encoding = locale.getpreferredencoding(False)
        return (b"".join(self._output)).decode(encoding)

    def _start(self, width, height):
        command = self.get_ffmpeg_command(width, height)
        logging.debug("command is %s", " ".join(command))
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=8 * 1024,
        )
        return proc

    def _collect_output(self):
        buf = self._ffmpeg.stdout.read()
        if buf:
            self._output.append(buf)

    def get_ffmpeg_command(self, width, height, quality=18):
        if os.name == "nt":
            FFMPEG_BIN = "ffmpeg.exe"  # on Windows
        else:
            FFMPEG_BIN = "ffmpeg"  # on Linux ans Mac OS

        command = [
            FFMPEG_BIN,
            "-y",  # overwrite output file if it exists
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-loglevel",
            "warning",  # minimal output
            "-s",
            str(width) + "x" + str(height),  # size of one frame
            "-pix_fmt",
            self.pix_fmt,
            "-r",
            str(self.fps),  # frames per second
            "-i",
            "-",  # The imput comes from a pipe
            "-an",  # Tells FFMPEG not to expect any audio
            "-c:v",
            self.codec,
            "-b:v",
            "1000k",
            "-preset",
            "veryfast",
            str(self.filename),
        ]
        return command
