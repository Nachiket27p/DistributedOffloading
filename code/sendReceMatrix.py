import socket
import logging
import sys
import zfpy
import struct
import numpy as np
from io import BytesIO

DEF_HEADER_SIZE = 128


def __mat_pack_frame(frame):
    f = BytesIO()
    np.savez(f, frame=frame)

    packet_size = len(f.getvalue())
    header = '{0}:'.format(packet_size)
    header = bytes(header.encode())  # prepend length of array

    out = bytearray()
    out += header

    f.seek(0)
    out += f.read()
    return out


def mat_send(sendSocket, frame, logger):
    if not isinstance(frame, np.ndarray):
        raise TypeError("input frame is not a valid numpy array")

    out = __mat_pack_frame(frame)

    try:
        sendSocket.sendall(out)
    except BrokenPipeError:
        logger.error("connection broken")
        raise

    logger.debug("frame sent")


def mat_receive(recieveSocket, logger, socket_buffer_size=1024):
    length = None
    frameBuffer = bytearray()
    while True:
        data = recieveSocket.recv(socket_buffer_size)
        frameBuffer += data
        if len(frameBuffer) == length:
            break
        while True:
            if length is None:
                if b':' not in frameBuffer:
                    break
                # remove the length bytes from the front of frameBuffer
                # leave any remaining bytes in the frameBuffer!
                length_str, ignored, frameBuffer = frameBuffer.partition(b':')
                length = int(length_str)
            if len(frameBuffer) < length:
                break
            # split off the full message from the remaining bytes
            # leave any remaining bytes in the frameBuffer!
            frameBuffer = frameBuffer[length:]
            length = None
            break

    frame = np.load(BytesIO(frameBuffer))['frame']
    logger.debug("frame received")
    return frame


def mat_send_comp(sendSocket, frame, logger):
    if not isinstance(frame, np.ndarray):
        raise TypeError("input frame is not a valid numpy array")

    sizeBefore = frame.size * frame.itemsize

    compFrame = zfpy.compress_numpy(frame)

    logger.info("Bytes: " + str(sizeBefore) + ' ---> ' + str(sys.getsizeof(compFrame)))

    data = struct.pack('>I', len(compFrame)) + compFrame

    try:
        sendSocket.sendall(data)
    except BrokenPipeError:
        logger.error("connection broken")
        raise

    logger.debug("frame sent")


def __mat_receive_all_comp(recieveSocket, n):
    data = bytearray()
    while len(data) < n:
        packet = recieveSocket.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)

    return data


def mat_receive_comp(recieveSocket, logger):
    rawDataLen = __mat_receive_all_comp(recieveSocket, 4)
    if not rawDataLen:
        return None

    dataLen = struct.unpack('>I', rawDataLen)[0]

    data = __mat_receive_all_comp(recieveSocket, dataLen)

    frame = zfpy.decompress_numpy(bytes(data))

    logger.debug("frame received")
    return frame
