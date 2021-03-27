import time
import socket
import os
import logging
import numpy as np
from sendReceCompMatrix import mat_send, mat_receive
from sendReceCompMatrix import DEF_HEADER_SIZE

# create logging
logger = logging.getLogger("Worker:" + str(os.getpid()))

# set up socket connection to main node
mainNode = socket.socket()
host = '127.0.0.1'
port = 5000

# used to log times
timePoints = [0, 0, 0, 0]
timePointLables = ['Time confirm header', 'Time taken to receive data', 'Time taken to compute', 'Time to send results']

# try to establish connection to main node
try:
    mainNode.connect((host, port))
    timePoints[0] = time.time_ns()
except socket.error as e:
    print(str(e))
    mainNode.close()
    exit()


taskHeader = mainNode.recv(DEF_HEADER_SIZE)
shape = taskHeader.decode('utf-8').split('|')

taskHeaderConf = shape[0] + '=' + shape[1] + 'x' + shape[2]

mainNode.send(str.encode(taskHeaderConf))

timePoints[1] = time.time_ns()
timePoints[0] = timePoints[1] - timePoints[0]

mat_a = mat_receive(mainNode, logger)
mainNode.send(str.encode(str(mat_a.shape)))
logger.info(mat_a)

mat_b = mat_receive(mainNode, logger)
mainNode.send(str.encode(str(mat_b.shape)))
logger.info(mat_b)

timePoints[2] = time.time_ns()
timePoints[1] = timePoints[2] - timePoints[1]

result = np.matmul(mat_a, mat_b)

timePoints[3] = time.time_ns()
timePoints[2] = timePoints[3] - timePoints[2]

mat_send(mainNode, result, logger)

tempTime = time.time_ns()
timePoints[3] = tempTime - timePoints[3]


print("Task Complete")
for tl in timePointLables:
    print(tl)
for t in timePoints:
    print(t)

mainNode.close()
