SOCKET_NAME = "/var/run/classifier"
import json
import sys
import socket
import argparse
import logging
from ml_tools.logs import init_logging
from config.config import Config
import time


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args(cmd_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Dont keep video frames in memory for classification later, but cache them to disk (Best for large videos, but slower)",
    )
    parser.add_argument(
        "-o",
        "--meta-to-stdout",
        action="count",
        help="Print metadata to stdout instead of saving to file.",
    )
    parser.add_argument(
        "source",
        help="a CPTV file to classify",
    )
    if cmd_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_args)
    return args


def main(cmd_args=None):
    args = parse_args(cmd_args)
    init_logging()
    logging.info("Classifying %s", args.source)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    config = Config.load_from_file()
    results = ""
    try:
        count = 0
        retries = 1
        while True:
            try:
                sock.connect(config.classify.service_socket)
                logging.info("Connected to %s ", config.classify.service_socket)
                break
            except Exception as ex:
                if count >= retries:
                    raise ex
                count += 1
                logging.warning(
                    "Could not connect to %s retrying in 10s error was %s",
                    config.classify.service_socket,
                    ex,
                )
                time.sleep(10)
        data = {"file": args.source, "cache": args.cache, "reuse_frames": False}
        sock.send(json.dumps(data).encode())

        results = read_all(sock).decode()
        meta_data = json.loads(str(results))
        if "error" in meta_data:
            raise Exception(meta_data["error"])

            logging.error("Error classifying %s %s", args.source, meta_data["error"])
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    finally:
        # Clean up the connection
        sock.close()
    if args.meta_to_stdout:
        print(str(results))
    return str(results)


def read_all(socket):
    size = 4096
    data = bytearray()

    while True:
        packet = socket.recv(size)
        if packet:
            data.extend(packet)
        else:
            break
    return data


if __name__ == "__main__":
    main()
