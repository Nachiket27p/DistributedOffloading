import socket
import logging
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


def mat_recieve(recieveSocket, logger, socket_buffer_size=1024):
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
