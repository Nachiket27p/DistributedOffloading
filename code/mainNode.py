
import socket
import logging
import os
from _thread import *
import numpy as np
import random as rnd
from sendReceMatrix import mat_send, mat_receive
from sendReceMatrix import mat_send_comp, mat_receive_comp
from sendReceMatrix import DEF_HEADER_SIZE
from workers import WorkerList
from threading import Thread, Lock, Semaphore

# worker list
wList = WorkerList()
# locks to protect the worker list
mutex = Lock()
full = Semaphore(0)

# create logging
logging.basicConfig(filename='logs/main.log', level=logging.DEBUG)
logger = logging.getLogger("Main:")
logging.debug("\n\n\n")

# split size
SPLIT = 4
# set up socket for this main node
mainSocket = socket.socket()
# host = '192.168.1.9'
host = '127.0.0.1'
port = 5000

ThreadCount = 0

r = 8
c = 8
rc = r * c
# mat_a = np.arange(rc).reshape(r, c)
# mat_b = np.arange(rc).reshape(r, c)
mat_a = (np.arange(rc).reshape(r, c)).astype(np.float)
mat_b = (np.arange(rc).reshape(r, c)).astype(np.float)

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
def threaded_client(worker, taskID, rows, cols):
    tlgr = logging.getLogger(taskID + ':')
    rowsShapeStr = str(rows.shape)
    colsShapeStr = str(cols.shape)
    taskHeader = taskID + '|' + rowsShapeStr + '|' + colsShapeStr
    taskHeaderConf = taskID + '=' + rowsShapeStr + 'x' + colsShapeStr
    tlgr.info('Task Header Sent: ' + taskHeader)

    worker.send(str.encode(taskHeader))

    taskHeaderResponse = (worker.recv(DEF_HEADER_SIZE)).decode('utf-8')

    tlgr.info('Task Header Response: ' + taskHeaderResponse)
    tlgr.info('Task Header Expected: ' + taskHeaderConf)

    if(taskHeaderConf == taskHeaderResponse):
        # mat_send(worker, rows, tlgr)
        mat_send_comp(worker, rows, tlgr)
        mat_a_rec = (worker.recv(DEF_HEADER_SIZE)).decode('utf-8')
        if(mat_a_rec != rowsShapeStr):
            tlgr.info(mat_a_rec)
            return
        tlgr.info('Worker Confirmed matrix a received')

        # mat_send(worker, cols, tlgr)
        mat_send_comp(worker, cols, tlgr)
        mat_b_rec = (worker.recv(DEF_HEADER_SIZE)).decode('utf-8')
        if(mat_b_rec != colsShapeStr):
            tlgr.info(mat_b_rec)
            return
        tlgr.info('Worker Confirmed matrix b received')

    else:
        tlgr.error('Task not received correctly')
        return

    # results = mat_recieve(worker, tlgr)
    results = mat_receive_comp(worker, tlgr)
    # log the results
    tlgr.info(results)

    # place the worker back into the free set of the worker list
    mutex.acquire()
    wList.freeWorker(worker)
    mutex.release()


def workerHandler():
    wlgr = logging.getLogger('WorkerHandler:')
    while True:
        # check if there are enough workers to offload the task
        # if there are then signal the main thread
        if wList.freeSize() >= SPLIT:
            full.release()

        # wait for new workers to join
        worker, address = mainSocket.accept()
        wlgr.info('Connected to: ' + address[0] + ':' + str(address[1]))
        mutex.acquire()
        wList.addWorker(worker)
        mutex.release()


def main():
    # wait for workers to connect
    start_new_thread(workerHandler, ())

    try:
        while True:
            # try to acquire the semaphore to indicate
            # enough workers have connected to the
            full.acquire()

            # acquire the mutex to get access to the worker list
            mutex.acquire()

            for i in range(SPLIT):
                start = len(mat_a) // SPLIT * i
                end = len(mat_a) // SPLIT * (i + 1)

                # get worker
                worker = wList.getWorker()
                # assign random id to task which will be used during communication
                subTaskID = str(rnd.randint(0, 1024))
                # start a worker thread
                start_new_thread(threaded_client, (worker, subTaskID, mat_a[start:end, :], mat_b[:, start:end]))

            # release the mutex to free the worker list
            mutex.release()

    except KeyboardInterrupt:
        logger.info('Shutting down server ...')

    mainSocket.close()


if __name__ == '__main__':
    main()
