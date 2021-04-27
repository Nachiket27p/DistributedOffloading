import socket
import os
import logging
from time import sleep
import numpy as np
from transportMM import send_mm, recv_mm
from transportMM import DEF_HEADER_SIZE

# set up socket connection to main node
mainSocket = socket.socket()

# configure the main node IP and local ip
hip = '10.142.0.2'
ip = socket.gethostbyname(socket.gethostname())
# hip = '127.0.0.1'
# ip = '127.0.0.1'

portI = 5005
portW = None

try:
    mainSocket.connect((hip, portI))
    msg = mainSocket.recv(DEF_HEADER_SIZE)
    portW = int(msg.decode('utf-8'))
    print('Connected')
except socket.error as e:
    print(e)
    mainSocket.close()
    exit()

# close the initial connection
mainSocket.close()

# sleep for 0.1 seconds for ports to be freed
sleep(0.1)

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
        # accept connection from main node for task
        mainNode, address = mainSocket.accept()

        # receive the task header
        # can be used to determine if something can be
        # not strictly necessary, can be removed
        rawTaskHeader = mainNode.recv(DEF_HEADER_SIZE)
        taskHeader = rawTaskHeader.decode('utf-8')
        print('Got work to do! :', taskHeader)

        # decode the task header
        task = taskHeader.split('|')
        taskHeaderConf = task[0] + '=' + task[1] + 'x' + task[2]

        # determine if the main node provided a tolerance level
        if task[3] == 'None':
            atol = None
        else:
            atol = float(atol)

        mainNode.send(str.encode(taskHeaderConf))

        # receive first matrix
        mat_a = recv_mm(mainNode)
        mainNode.send(str.encode(str(mat_a.shape)))

        # receive second matrix
        mat_b = recv_mm(mainNode)
        mainNode.send(str.encode(str(mat_b.shape)))

        print('Got both matrices')

        # confirm the main node is ready to receive results
        # send the results to main node
        sendResConf = mainNode.recv(DEF_HEADER_SIZE)
        result = np.matmul(mat_a, mat_b)
        send_mm(mainNode, result)

        # close the connection after completing task
        mainNode.close()

except KeyboardInterrupt:
    print("Shutting down worker ...")

mainSocket.close()
