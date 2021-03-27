import socket
import logging
from struct import pack
import struct
import numpy as np
from io import BytesIO
import zfpy

DEF_HEADER_SIZE = 128


def mat_send(sendSocket, frame, logger):
    if not isinstance(frame, np.ndarray):
        raise TypeError("input frame is not a valid numpy array")

    compFrame = zfpy.compress_numpy(frame)

    data = struct.pack('>I', len(compFrame)) + compFrame

    try:
        sendSocket.sendall(data)
    except BrokenPipeError:
        logger.error("connection broken")
        raise

    logger.debug("frame sent")


def __mat_receive_all(recieveSocket, n):
    data = bytearray()
    while len(data) < n:
        packet = recieveSocket.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)

    return data


def mat_receive(recieveSocket, logger):
    rawDataLen = __mat_receive_all(recieveSocket, 4)
    if not rawDataLen:
        return None

    dataLen = struct.unpack('>I', rawDataLen)[0]

    data = __mat_receive_all(recieveSocket, dataLen)

    frame = zfpy.decompress_numpy(bytes(data))

    logger.debug("frame received")
    return frame
