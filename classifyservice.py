import threading
import socket
from config.config import Config
from ml_tools.logs import init_logging
from classify.clipclassifier import ClipClassifier
import logging
import os
import argparse
import json
import traceback
from ml_tools.tools import CustomJSONEncoder


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
    try:
        service.run()
    except KeyboardInterrupt:
        logging.info("Keyboard interupt closing down")
    except PermissionError:
        logging.error("Error with permissions", exc_info=True)
    except:
        logging.error("Error with service restarting", exc_info=True)


class ClassifyService:
    def __init__(self, config):
        self.config = config

        self.clip_classifier = ClipClassifier(
            config,
        )

    def run(self):
        logging.info("Running on %s", self.config.classify.service_socket)
        max_jobs = 2
        try:
            os.unlink(self.config.classify.service_socket)
        except OSError:
            if os.path.exists(self.config.classify.service_socket):
                raise

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(self.config.classify.service_socket)
        sock.listen(1)
        self.clip_classifier.models = {}
        self.clip_classifier.load_models()

        while True:
            logging.info("waiting for jobs")
            connection, client_address = sock.accept()
            t = threading.Thread(
                target=classify_job,
                args=(self.clip_classifier, connection, client_address),
            )
            t.start()


def classify_job(clip_classifier, clientsocket, addr):
    try:
        job = read_all(clientsocket).decode()
        args = json.loads(job)
        if "file" not in args:
            logging.error("File name must be specified in argument dictionary")
            clientsocket.sendall(
                json.dumps(
                    {"error": "File name must be specified in argument dictionary"}
                ).encode()
            )
            return
        logging.info("Classifying %s with args %s", args["file"], args)

        meta_data = clip_classifier.process_file(
            args["file"],
            cache=args.get("cache"),
            reuse_frames=args.get("reuse_frames"),
        )

        response = clientsocket.sendall(
            json.dumps(meta_data, cls=CustomJSONEncoder).encode()
        )
        if response:
            logging.error("Error sending data to socket %s", response)
    except BrokenPipeError:
        logging.error(
            "Error sending metadata for job %s too %s",
            args["file"],
            addr,
            exc_info=True,
        )
    except Exception as e:
        logging.error("Error classifying job %s", args["file"], exc_info=True)
        clientsocket.sendall(
            json.dumps(
                {"error": f"Error classifying {traceback.format_exc()}"}
            ).encode()
        )
        raise e
    finally:
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
