
import socket
import logging
import os
import time
from _thread import *
import numpy as np
import random as rnd
from sendReceMatrix import mat_send, mat_receive
from sendReceMatrix import mat_send_comp, mat_receive_comp
from sendReceMatrix import DEF_HEADER_SIZE
from workers import WorkerList
from threading import Thread, Lock, Semaphore
from math import sqrt

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
portI = 5000
portW = 5001

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
    mainSocket.bind((host, portI))
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
def threaded_client(workerID, taskID, rows, cols, subResults):
    # initialize logger for this thread
    tlgr = logging.getLogger(str(taskID) + ':')
    # connect to worker node
    worker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # try to establish connection to worker
    try:
        worker.connect(workerID)
    except socket.error as e:
        worker.close()
        tlgr.error('connection to ' + str(workerID) + ' failed with: ' + str(e))
        exit()

    rowsShapeStr = str(rows.shape)
    colsShapeStr = str(cols.shape)
    taskHeader = str(taskID) + '|' + rowsShapeStr + '|' + colsShapeStr
    taskHeaderConf = str(taskID) + '=' + rowsShapeStr + 'x' + colsShapeStr
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
        worker.send(str.encode('start work!'))
    else:
        tlgr.error('Task not received correctly')
        return

    # results = mat_recieve(worker, tlgr)
    results = mat_receive_comp(worker, tlgr)
    subResults[taskID[0]][taskID[1]] = results
    # log the results
    tlgr.info(results)

    # place the worker back into the free set of the worker list
    mutex.acquire()
    wList.freeWorker(workerID)
    mutex.release()


def workerHandler():
    wlgr = logging.getLogger('WorkerHandler:')
    while True:
        # wait for new workers to join
        worker, address = mainSocket.accept()
        wlgr.info('Connected to: ' + address[0] + ':' + str(address[1]))
        worker.send(str.encode(str(address[1])))
        mutex.acquire()
        wList.addWorker(address)
        mutex.release()
        worker.close()


def main():
    # wait for workers to connect
    start_new_thread(workerHandler, ())

    try:
        while True:
            tries = 0
            while True:
                # waited (__timeOut * __tries) seconds with no success, then raise exception
                if tries > 5:
                    mainSocket.close()
                    logger.error('Minimum number workers (' + str(SPLIT + 1) + ') not available.')
                    exit()
                mutex.acquire()  # acquire lock for '__wList' variable

                if wList.freeSize() > SPLIT:
                    # if there are enough workers then break out of this loop
                    mutex.release()  # release lock for '__wList'  variable
                    break
                elif (wList.occupiedSize() + wList.freeSize()) > SPLIT:
                    # if there are enough workers but they are busy then reset the tries because
                    # the task can be distributed at some point in the future
                    tries = 0

                mutex.release()  # release lock for '__wList'  variable
                # if there are not enough workers then sleep for a couple of seconds
                tries += 1
                time.sleep(2)

            # acquire the mutex to get access to the worker list
            mutex.acquire()

            # squre root of the numbers of tasks to evenly split tasks
            SSplit = int(sqrt(SPLIT))

            subResults = [[None] * SSplit for _ in range(SSplit)]
            wThreads = []

            for i in range(SSplit):
                # compute the start and end row for the sub task for matrix a
                sR = (len(mat_a) // SSplit) * i
                eR = sR + (len(mat_a) // SSplit)
                for j in range(SSplit):
                    # compute the start and end column for the sub task for matrix b
                    sC = (len(mat_b[0]) // SSplit) * j
                    eC = sC + (len(mat_b[0]) // SSplit)
                    # get worker
                    worker = wList.getWorker()
                    # assign random id to task which will be used during communication
                    subTaskID = (i, j)
                    # start a worker thread
                    wt = Thread(target=threaded_client, args=(worker, subTaskID, mat_a[sR:eR, :], mat_b[:, sC:eC], subResults))
                    wt.start()
                    wThreads.append(wt)
                    # start_new_thread(threaded_client, (worker, subTaskID, mat_a[sR:eR, :], mat_b[:, sC:eC], subResults))

            # release the mutex to free the worker list
            mutex.release()

            for t in wThreads:
                t.join()

            logger.info(np.bmat(subResults))

            while True:
                pass

    except KeyboardInterrupt:
        logger.info('Shutting down server ...')

    mainSocket.close()


if __name__ == '__main__':
    main()
