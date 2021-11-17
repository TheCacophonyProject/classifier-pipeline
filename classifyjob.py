SOCKET_NAME = "/var/run/classifier"
import json
import sys
import socket
from config.config import Config


def main(cptv_file):
    print("Classifying", cptv_file)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    config = Config.load_from_file()

    try:
        sock.connect(config.classify.service_socket)
        data = {"file": cptv_file, "cache": True, "reuse_frames": False}
        sock.send(json.dumps(data).encode())

        results = read_all(sock).decode()
        meta_data = json.loads(str(results))
        if "error" in meta_data:
            raise Exception(meta_data["error"])

            print("ERRR", meta_data["error"])
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    finally:
        # Clean up the connection
        sock.close()


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
    main(sys.argv[1])
