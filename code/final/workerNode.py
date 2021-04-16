import socket
import os
import logging
import numpy as np
from transportMM import send_mm, recv_mm
from transportMM import DEF_HEADER_SIZE

# set up socket connection to main node
mainNode = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# host = '192.168.1.9'
host = '127.0.0.1'
port = 5000

try:
    mainNode.connect((host, port))
except socket.error as e:
    mainNode.close()
    print(str(e))
    exit()

# wait for workers to connect
try:
    # if no exception was thrown then wait for work
    while True:
        taskHeader = mainNode.recv(DEF_HEADER_SIZE)
        if(taskHeader == b''):
            continue
        shape = taskHeader.decode('utf-8').split('|')

        taskHeaderConf = shape[0] + '=' + shape[1] + 'x' + shape[2]

        mainNode.send(str.encode(taskHeaderConf))

        mat_a = recv_mm(mainNode)
        mainNode.send(str.encode(str(mat_a.shape)))

        mat_b = recv_mm(mainNode)
        mainNode.send(str.encode(str(mat_b.shape)))

        result = np.matmul(mat_a, mat_b)
        send_mm(mainNode, result)

except KeyboardInterrupt:
    print("Shutting down worker ...")

mainNode.close()
