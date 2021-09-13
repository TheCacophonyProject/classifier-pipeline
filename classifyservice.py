SOCKET_NAME = "/var/run/classifier"
import threading
import socket
from config.config import Config
from ml_tools.logs import init_logging
from classify.clipclassifier import ClipClassifier
import pickle
import logging
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config-file", help="Path to config file to use")
    return parser.parse_args()


# Links to socket and continuously waits for 1 connection
def main():
    init_logging()
    args = parse_args()

    config = Config.load_from_file(args.config_file)
    service = ClassifyService(config)
    while True:
        try:
            service.run()
        except KeyboardInterrupt:
            logging.info("Keyboard interupt closing down")
            break
        except:
            logging.error("Error with service restarting", exc_info=True)


class ClassifyService:
    def __init__(self, config):
        self.config = config

        self.clip_classifier = ClipClassifier(
            config,
        )
        self.clip_classifier.load_models()

    def run(self):
        max_jobs = 2
        try:
            os.unlink(SOCKET_NAME)
        except OSError:
            if os.path.exists(SOCKET_NAME):
                raise

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(SOCKET_NAME)
        sock.listen(1)
        while True:
            logging.info("waiting for a connection")
            connection, client_address = sock.accept()
            t = threading.Thread(
                target=classify_job,
                args=(self.clip_classifier, connection, client_address),
            )
            t.start()


def classify_job(clip_classifier, clientsocket, addr):
    try:
        job = read_all(clientsocket)
        args = pickle.loads(job)
        if "file" not in args:
            logging.error("File name must be specified in argument dictionary")
        logging.info("Classifying %s with cache? %s", args["file"], args)
        meta_data = clip_classifier.process_file(
            args["file"],
            cache=args.get("cache"),
            resuse_frames=args.get("resuse_frames"),
        )
        clientsocket.send(pickle.dumps(meta_data))

    except:
        logging.error(
            "Error sending metadata for job %s too %s", args[0], addr, exc_info=True
        )
    try:
        clientsocket.close()
    except:
        pass


def read_all(socket):
    size = 4096
    data = bytearray()

    while size > 0:
        packet = socket.recv(size)
        data.extend(packet)
        if len(packet) < size:
            break
    return data


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    main()
