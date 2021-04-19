import socket
import zfpy
import struct
import numpy as np
from threading import Lock

DEF_HEADER_SIZE = 128
cacheMutex = Lock()


def send_mm(sendSocket, frame, compDataCache=None, key=None):
    """USed to send numpy matrix data using zfpy compression module.
    If the main node is distributing the work matrices, then a caching
    mechanism can be used to keep track of already compressed data.

    Args:
        sendSocket (socket): The socket connection on which the data is being sent
        frame (ndarray): The numpy matrix being sent
        compDataCache (dict, optional): A dictionary which represents a caching mechanism
                                        Defaults to None.
        key (any, optional): The key can generally be anything, the correct use of the key is left to the user
                                Defaults to None.

    Raises:
        TypeError: Raised if the data provided in not a numpy array
        Exception: Raised if the connection is broken during transmission of the data
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError("input frame is not a valid numpy array")

    data = None
    # if the main node is sending the matrix data then
    # use the caching mechanism inplace in the event of a failed
    # worker node, data does not need to be recompressed
    if not(compDataCache == None and key == None):
        needToCompress = False
        cacheMutex.acquire()
        if key not in compDataCache:
            needToCompress = True
        cacheMutex.release()

        # compress if the key could not be found initially
        if needToCompress:
            compFrame = zfpy.compress_numpy(frame)
            # protect cache data structure
            cacheMutex.acquire()
            compDataCache[key] = compFrame
            cacheMutex.release()

        cacheMutex.acquire()
        data = struct.pack('>I', len(compDataCache[key])) + compDataCache[key]
        cacheMutex.release()
    else:
        compFrame = zfpy.compress_numpy(frame)
        data = struct.pack('>I', len(compFrame)) + compFrame

    # try to send all the data packet constructed
    try:
        sendSocket.sendall(data)
    except BrokenPipeError:
        cacheMutex.release()
        raise Exception('Connection broken!')


def __recv_all_mm(recieveSocket, n):
    """
    Private function to this file, should not be used from outside this file unless
    you are aware of how it works.
    This function is called by recv_mm to receive the compressed numpy data.

    Args:
        recieveSocket (socket): The socket on which the data is being received
        n (int): The size of the data still remaining to be

    Returns:
        struct: The compresse numpy data
    """
    data = bytearray()
    while len(data) < n:
        packet = recieveSocket.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)

    return data


def recv_mm(recieveSocket):
    """
    Used to receive compressed numpy data on the socket provided.
    This function calls the __recv_all_mm() to achieve this to retrieve the
    compressed data. Once receiving the compressed data enclosed in a struct,
    zfpy uncompresses uncompresses the data.

    Args:
        recieveSocket (socket): The socket on which the data is being received

    Returns:
        ndarray: The uncompressed numpy array
    """
    rawDataLen = __recv_all_mm(recieveSocket, 4)
    if not rawDataLen:
        return None

    dataLen = struct.unpack('>I', rawDataLen)[0]

    data = __recv_all_mm(recieveSocket, dataLen)

    frame = zfpy.decompress_numpy(bytes(data))

    return frame
