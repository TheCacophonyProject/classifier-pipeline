import fcntl
import locale
import os
import subprocess


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
    def __init__(self, filename, quality=21):
        self.filename = filename
        self.quality = quality
        self._ffmpeg = None
        self._output = []

    def next_frame(self, frame):
        if self._ffmpeg is None:
            height, width, _ = frame.shape
            self._ffmpeg = self._start(width, height)

        self._collect_output()
        self._ffmpeg.stdin.write(frame.tobytes())

    def close(self):
        self._collect_output()
        self._ffmpeg.stdin.close()

        return_code = self._ffmpeg.wait(timeout=60)
        if return_code:
            self._collect_output()
            raise Exception("ffmpeg failed with error {}. output:\n{}".format(
                return_code, self.output))

    @property
    def output(self):
        encoding = locale.getpreferredencoding(False)
        return (b"".join(self._output)).decode(encoding)

    def _start(self, width, height):
        command = get_ffmpeg_command(self.filename, width, height,
                                     self.quality)
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=4096)
        make_non_blocking(proc.stdout)
        return proc

    def _collect_output(self):
        buf = self._ffmpeg.stdout.read()
        if buf:
            self._output.append(buf)


def get_ffmpeg_command(filename, width, height, quality=21):
    if os.name == 'nt':
        FFMPEG_BIN = "ffmpeg.exe"  # on Windows
    else:
        FFMPEG_BIN = "ffmpeg"  # on Linux ans Mac OS

    command = [
        FFMPEG_BIN,
        '-y',  # overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-loglevel', 'warning', # minimal output
        '-s', str(width) + 'x' + str(height),  # size of one frame
        '-pix_fmt', 'rgb24',
        '-r', '9',  # frames per second
        '-i', '-',  # The imput comes from a pipe
        '-an',  # Tells FFMPEG not to expect any audio
        '-vcodec', 'libx264',
        '-tune', 'grain',  # good for keepinn the grain in our videos
        '-crf', str(quality),  # quality, lower is better
        '-pix_fmt', 'yuv420p',  # window thumbnails require yuv420p for some reason
        filename
    ]
    return command


def make_non_blocking(f):
    fl = fcntl.fcntl(f, fcntl.F_GETFL)
    fcntl.fcntl(f, fcntl.F_SETFL, fl | os.O_NONBLOCK)
