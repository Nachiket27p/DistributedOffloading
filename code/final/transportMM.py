import socket
import zfpy
import struct
import numpy as np

DEF_HEADER_SIZE = 128


def send_mm(sendSocket, frame):
    if not isinstance(frame, np.ndarray):
        raise TypeError("input frame is not a valid numpy array")

    compFrame = zfpy.compress_numpy(frame)

    data = struct.pack('>I', len(compFrame)) + compFrame

    try:
        sendSocket.sendall(data)
    except BrokenPipeError:
        raise Exception('Connection broken!')


def __recv_all_mm(recieveSocket, n):
    data = bytearray()
    while len(data) < n:
        packet = recieveSocket.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)

    return data


def recv_mm(recieveSocket):
    rawDataLen = __recv_all_mm(recieveSocket, 4)
    if not rawDataLen:
        return None

    dataLen = struct.unpack('>I', rawDataLen)[0]

    data = __recv_all_mm(recieveSocket, dataLen)

    frame = zfpy.decompress_numpy(bytes(data))

    return frame
