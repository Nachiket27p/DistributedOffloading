
import socket
import logging
import os
from _thread import *
import numpy as np
import random as rnd
from sendReceMatrix import mat_send, mat_recieve
from sendReceMatrix import DEF_HEADER_SIZE


# create logging
logging.basicConfig(filename='logs/main.log', level=logging.DEBUG)
logger = logging.getLogger("Main Node: " + str(os.getpid()))
logging.debug("\n\n\n")

# set up socket for this main node
mainSocket = socket.socket()
host = '127.0.0.1'
port = 1234
ThreadCount = 0

mat_a = np.arange(100).reshape(10, 10)
mat_b = np.arange(100).reshape(10, 10)

# check if port is available
try:
    mainSocket.bind((host, port))
except socket.error as e:
    logger.error(str(e))

maxServQ = 5
minWorkers = 4

print('Server Started ...')
logger.info('Server Started ...')

# max number of queued connections before refusing
mainSocket.listen(maxServQ)

logger.info('Waitiing for minimum of ' + str(minWorkers) + ' workers to connect..')


# used to send work to worker
def threaded_client(connection, taskID, rows, cols):
    taskHeader = taskID + '|' + str(rows.shape) + '|' + str(cols.shape)
    taskHeaderConf = taskID + '=' + str(rows.shape) + 'x' + str(cols.shape)
    logger.info('Task Header Sent: ' + taskHeader)

    connection.send(str.encode(taskHeader))

    taskHeaderResponse = (connection.recv(DEF_HEADER_SIZE)).decode('utf-8')

    logger.info('Task Header Response: ' + taskHeaderResponse)
    logger.info('Task Header Expected: ' + taskHeaderConf)

    if(taskHeaderConf == taskHeaderResponse):
        mat_send(connection, rows, logger)
        mat_send(connection, cols, logger)
    else:
        logger.error("Task not received correctly")
        return

    results = mat_recieve(connection, logger)

    logger.info(results)


# wait for workers to connect
try:
    while True:
        worker, address = mainSocket.accept()

        logger.info('Connected to: ' + address[0] + ':' + str(address[1]))

        subTaskID = str(rnd.randint(0, 1024))

        start_new_thread(threaded_client, (worker, subTaskID, mat_a[:10, :], mat_b[:, :10]))

        ThreadCount += 1

        logger.info('Thread Number: ' + str(ThreadCount))

except KeyboardInterrupt:
    logger.info('Shutting down server ...')


mainSocket.close()
