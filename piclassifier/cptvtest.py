#!/usr/bin/python3

import socket
import time
import os
from cptv import CPTVReader
import sys
import numpy as np

SOCKET_NAME = "/var/run/lepton-frames"
test_cptv = "test.cptv"


def send_cptv(filename, socket):
    """
    Loads a cptv file, and prepares for track extraction.
    """
    with open(filename, "rb") as f:
        reader = CPTVReader(f)
        for i, frame in enumerate(reader):
            f = np.uint16(frame.pix).byteswap()
            socket.sendall(f)
            print("sending frame {}".format(i))


sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)

try:
    sock.connect(SOCKET_NAME)
    print("sending cptv file")
    send_cptv(test_cptv, sock)
except socket.error as msg:
    print(msg)
    sys.exit(1)
finally:
    # Clean up the connection
    sock.close()
