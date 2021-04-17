import socket
import os
import logging
import numpy as np
from sendReceMatrix import mat_send, mat_receive
from sendReceMatrix import mat_send_comp, mat_receive_comp
from sendReceMatrix import DEF_HEADER_SIZE

# create logging
logging.basicConfig(filename='logs/worker.log', level=logging.DEBUG)
logger = logging.getLogger("Worker:" + str(os.getpid()))
logging.debug("\n\n\n")

# set up socket connection to main node
mainSocket = socket.socket()
# host = '192.168.1.9'
ip = '127.0.0.1'
portI = 5003
portW = None


# try to establish connection to main node
logger.info('Requesting connection to main node.')
try:
    mainSocket.connect((ip, portI))
    msg = mainSocket.recv(DEF_HEADER_SIZE)
    portW = int(msg.decode('utf-8'))
    logger.info('Established connection to main node @ ' + ip + ':' + str(portW))
except socket.error as e:
    logger.error(str(e))
    mainSocket.close()
    exit()

# close the initial connection
mainSocket.close()
# wait for work from main node
mainSocket = socket.socket()
try:
    mainSocket.bind((ip, portW))
except socket.error as e:
    mainSocket.close()
    logger.info("Failed to bind (ip, port) to socket.")
    exit()

# define the number of request waiting in queue before rejecting connection request
maxConQ = 3
mainSocket.listen(maxConQ)

# wait for workers to connect
try:
    # if no exception was thrown then wait for work
    while True:
        logger.info('Waiting for task')
        mainNode, address = mainSocket.accept()

        taskHeader = mainNode.recv(DEF_HEADER_SIZE)
        shape = taskHeader.decode('utf-8').split('|')

        taskHeaderConf = shape[0] + '=' + shape[1] + 'x' + shape[2]
        logger.info(taskHeaderConf)

        mainNode.send(str.encode(taskHeaderConf))

        # mat_a = mat_recieve(mainNode, logger)
        mat_a = mat_receive_comp(mainNode, logger)
        mainNode.send(str.encode(str(mat_a.shape)))
        logger.info(mat_a)

        # mat_b = mat_recieve(mainNode, logger)
        mat_b = mat_receive_comp(mainNode, logger)
        mainNode.send(str.encode(str(mat_b.shape)))
        logger.info(mat_b)

        confMsg = mainNode.recv(DEF_HEADER_SIZE)
        if('start work!' != confMsg.decode('utf-8')):
            raise KeyboardInterrupt

        result = np.matmul(mat_a, mat_b)

        logger.info(result)

        # mat_send(mainNode, result, logger)
        mat_send_comp(mainNode, result, logger)

        # close the connection after completing task
        mainNode.close()

except KeyboardInterrupt:
    logger.info("Shutting down worker ...")

# close the listening socket
mainSocket.close()
