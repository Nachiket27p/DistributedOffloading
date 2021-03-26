import socket
import os
import logging
import numpy as np
from sendReceMatrix import mat_send, mat_recieve
from sendReceMatrix import DEF_HEADER_SIZE

# create logging
logging.basicConfig(filename='logs/worker.log', level=logging.DEBUG)
logger = logging.getLogger("Worker:" + str(os.getpid()))
logging.debug("\n\n\n")

# set up socket connection to main node
mainNode = socket.socket()
host = '127.0.0.1'
port = 1234


# try to establish connection to main node
logger.info('Requesting connection to main node.')
try:
    mainNode.connect((host, port))
    logger.info("Established connection to main node.")
except socket.error as e:
    logger.error(str(e))
    mainNode.close()
    exit()

# wait for workers to connect
try:
    # if no exception was thrown then wait for work
    while True:
        logger.info('Waiting for task')

        taskHeader = mainNode.recv(DEF_HEADER_SIZE)
        shape = taskHeader.decode('utf-8').split('|')

        taskHeaderConf = shape[0] + '=' + shape[1] + 'x' + shape[2]
        logger.info(taskHeaderConf)

        mainNode.send(str.encode(taskHeaderConf))

        mat_a = mat_recieve(mainNode, logger)
        mat_b = mat_recieve(mainNode, logger)

        logger.info(mat_a)
        logger.info(mat_b)

        result = np.matmul(mat_a, mat_b)

        logger.info(result)

        mat_send(mainNode, result, logger)

except KeyboardInterrupt:
    logger.info("Shutting down worker ...")

mainNode.close()