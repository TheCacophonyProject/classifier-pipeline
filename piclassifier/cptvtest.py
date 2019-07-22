#!/usr/bin/python3

import socket
import time
import os
from cptv import CPTVReader

SOCKET_NAME = "/var/run/lepton-frames"
test_cptv = "test.cptv"


def send_cptv(filename, connection):
    """
    Loads a cptv file, and prepares for track extraction.
    """
    with open(filename, "rb") as f:
        reader = CPTVReader(f)
        for i, frame in enumerate(reader):
            connection.sendall(frame.pix)
            print(f"sending frame {i}")


# Make sure the socket does not already exist
try:
    os.unlink(SOCKET_NAME)
except OSError:
    if os.path.exists(SOCKET_NAME):
        raise

# Create a UDS socket
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

# Bind the socket to the address
print("starting up on {}".format(SOCKET_NAME))
sock.bind(SOCKET_NAME)

# Listen for incoming connections
sock.listen(1)

while True:
    # Wait for a connection
    print("waiting for a connection")
    connection, client_address = sock.accept()
    try:
        print("connection from", client_address)

        # Receive the data in small chunks and retransmit it
        # while True:
        print("sending cptv file")
        send_cptv(test_cptv, connection)
        # time.sleep(20)

    finally:
        # Clean up the connection
        connection.close()
