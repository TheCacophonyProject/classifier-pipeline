SOCKET_NAME = "/var/run/classifier"
import json
import sys
import socket
import argparse
import logging
from ml_tools.logs import init_logging
import time
from classifyservice import ClassifyJob


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
        "--service_socket",
        default="/etc/cacophony/thermal-classifier",
        help="Socket name",
    )

    parser.add_argument(
        "--ready",
        action="store_true",
        default=False,
        help="Check if classify service is up and running",
    )

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
        "-t",
        "--track",
        action="store_true",
        help="Run tracking on the file before extracting",
    )
    parser.add_argument(
        "--calculate-thumbnails",
        action="store_true",
        help="Calculate thumbnails",
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


def test_socket(sock, address):
    sock.connect(address)
    return True


def main(cmd_args=None):
    args = parse_args(cmd_args)
    init_logging()
    logging.info("Classifying %s", args.source)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if args.ready:
        return test_socket(sock, args.service_socket)
    results = ""
    try:
        count = 0
        retries = 1
        while True:
            try:
                sock.connect(args.service_socket)
                logging.info("Connected to %s ", args.service_socket)
                break
            except Exception as ex:
                if count >= retries:
                    raise ex
                count += 1
                logging.warning(
                    "Could not connect to %s retrying in 10s error was %s",
                    args.service_socket,
                    ex,
                )
                time.sleep(10)
        data = ClassifyJob(
            file=args.source,
            cache=args.cache,
            track=args.track,
            calculate_thumbnails=args.calculate_thumbnails,
        )
        sock.send(json.dumps(data.as_dict()).encode())

        results = read_all(sock).decode()
        meta_data = json.loads(str(results))
        if "error" in meta_data:
            logging.error("Error classifying %s %s", args.source, meta_data["error"])
            raise Exception(meta_data["error"])

    except socket.error as msg:
        print(msg)
        sys.exit(1)
    finally:
        # Clean up the connection
        sock.close()
    if args.meta_to_stdout:
        print(str(results))
    # else:
    #     source_file = Path(args.source)
    #     meta_filename = source_file.with_suffix(".txt")

    #     logging.info("saving meta data %s", meta_filename)
    #     with open(meta_filename, "w") as f:
    #         json.dump(meta_data, f, indent=4, cls=CustomJSONEncoder)


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
