SOCKET_NAME = "/var/run/classifier"
import pickle
import sys
import socket


def main(cptv_file):
    print("clasisfying", cptv_file)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    try:
        sock.connect(SOCKET_NAME)
        data = {"file": cptv_file}
        sock.send(pickle.dumps(data))

        results = read_all(sock)
        meta_data = pickle.loads(results)
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    finally:
        # Clean up the connection
        sock.close()


def read_all(socket):
    size = 4096
    data = bytearray()

    while size > 0:
        packet = socket.recv(size)
        data.extend(packet)
        if len(packet) < size:
            break
    return data


if __name__ == "__main__":
    main(sys.argv[1])
