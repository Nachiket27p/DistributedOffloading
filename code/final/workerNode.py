import socket
import os
import logging
from time import sleep
import numpy as np
from transportMM import send_mm, recv_mm
from transportMM import DEF_HEADER_SIZE

# set up socket connection to main node
mainSocket = socket.socket()
# host = '192.168.1.9'
ip = '127.0.0.1'
portI = 5000
portW = None

try:
    mainSocket.connect((ip, portI))
    msg = mainSocket.recv(DEF_HEADER_SIZE)
    portW = int(msg.decode('utf-8'))
except socket.error as e:
    print(e)
    mainSocket.close()
    exit()

# sleep for 0.2 seconds for ports to be freed
sleep(0.1)

# close the initial connection
mainSocket.close()
# wait for work from main node
mainSocket = socket.socket()
try:
    mainSocket.bind((ip, portW))
except socket.error as e:
    mainSocket.close()
    print('Failed to bind ' + str((ip, portW)) + ' to socket with error: ' + str(e))
    exit()

# define the number of request waiting in queue before rejecting connection request
maxConQ = 3
mainSocket.listen(maxConQ)

# wait for workers to connect
try:
    # if no exception was thrown then wait for work
    while True:
        mainNode, address = mainSocket.accept()

        taskHeader = mainNode.recv(DEF_HEADER_SIZE)
        # if(taskHeader == b''):
        #     continue
        shape = taskHeader.decode('utf-8').split('|')

        taskHeaderConf = shape[0] + '=' + shape[1] + 'x' + shape[2]

        mainNode.send(str.encode(taskHeaderConf))
        mat_a = recv_mm(mainNode)
        sleep(2)
        mainNode.send(str.encode(str(mat_a.shape)))

        mat_b = recv_mm(mainNode)
        sleep(1)
        mainNode.send(str.encode(str(mat_b.shape)))

        sendResConf = mainNode.recv(DEF_HEADER_SIZE)
        result = np.matmul(mat_a, mat_b)
        send_mm(mainNode, result)

        # close the connection after completing task
        mainNode.close()

except KeyboardInterrupt:
    print("Shutting down worker ...")

mainSocket.close()
