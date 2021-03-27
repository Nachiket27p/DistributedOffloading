import time
import socket
import logging
import os
from _thread import *
import numpy as np
import random as rnd
from sendReceCompMatrix import mat_send, mat_receive
from sendReceCompMatrix import DEF_HEADER_SIZE


# create logging
logger = logging.getLogger("Main Node: " + str(os.getpid()))

# set up socket for this main node
mainSocket = socket.socket()
host = '127.0.0.1'
port = 5000

ThreadCount = 0

mat_a = (np.arange(10000).reshape(100, 100)).astype(np.float)
mat_b = (np.arange(10000).reshape(100, 100)).astype(np.float)

mat_c = np.matmul(mat_a, mat_b)

# check if port is available
try:
    mainSocket.bind((host, port))
except socket.error as e:
    print(str(e))

maxServQ = 5
minWorkers = 4

print('Server Started ...')

# max number of queued connections before refusing
mainSocket.listen(maxServQ)

# used to log times
timePoints = [0, 0, 0, 0]
timePointLables = ['Time to start worker handler', 'Time for header comminication', 'Time taken to send data', 'Time taken to receive results']
# used to send work to worker


def threaded_client(connection, taskID, rows, cols):
    rowsShapeStr = str(rows.shape)
    colsShapeStr = str(cols.shape)

    timePoints[1] = time.time_ns()
    timePoints[0] = timePoints[1] - timePoints[0]

    taskHeader = taskID + '|' + rowsShapeStr + '|' + colsShapeStr
    taskHeaderConf = taskID + '=' + rowsShapeStr + 'x' + colsShapeStr

    connection.send(str.encode(taskHeader))

    taskHeaderResponse = (connection.recv(DEF_HEADER_SIZE)).decode('utf-8')

    if(taskHeaderConf == taskHeaderResponse):
        # measure the time taken for header transfer
        timePoints[2] = time.time_ns()
        timePoints[1] = timePoints[2] - timePoints[1]

        mat_send(connection, rows, logger)
        mat_a_rec = (connection.recv(DEF_HEADER_SIZE)).decode('utf-8')
        if(mat_a_rec != rowsShapeStr):
            logger.info(mat_a_rec)
            return

        mat_send(connection, cols, logger)
        mat_b_rec = (connection.recv(DEF_HEADER_SIZE)).decode('utf-8')
        if(mat_b_rec != colsShapeStr):
            logger.info(mat_b_rec)
            return

        # measure time taken to send matrices over
        timePoints[3] = time.time_ns()
        timePoints[2] = timePoints[3] - timePoints[2]
    else:
        return

    results = mat_recieve(connection, logger)

    tempTime = time.time_ns()
    timePoints[3] = tempTime - timePoints[3]

    print(results == mat_c)
    # print(results)


# wait for workers to connect
try:
    while True:
        worker, address = mainSocket.accept()
        subTaskID = str(rnd.randint(0, 1024))
        timePoints[0] = time.time_ns()
        start_new_thread(threaded_client, (worker, subTaskID, mat_a, mat_b))
        ThreadCount += 1

except KeyboardInterrupt:
    print()
    for tl in timePointLables:
        print(tl)
    for t in timePoints:
        print(t)


mainSocket.close()
