import os
import socket
import time
import random as rnd
import numpy as np
from math import sqrt
from _thread import *
from threading import Thread, Lock, Semaphore
from transportMM import send_mm, recv_mm
from transportMM import DEF_HEADER_SIZE
from workerList import WorkerList


class DMM:

    def __init__(self, hostIP='127.0.0.1', port=5000, maxListenQ=5, taskSplit=4, tries=5, sleepTime=2.0, offloadAttempts=1) -> None:
        self.__hostIP = hostIP
        self.__port = port
        self.__mainSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__maxListenQ = maxListenQ
        self.__taskSplit = taskSplit
        self.__tries = tries
        self.__sleepTime = sleepTime
        self.__offloadAttempts = offloadAttempts

        try:
            self.__mainSocket.bind((self.__hostIP, self.__port))
        except socket.error as e:
            raise Exception(str(e))

        self.__mainSocket.listen(self.__maxListenQ)

        # worker list
        self.__wList = WorkerList()
        # locks to protect the worker list
        self.__mutex = Lock()
        self.__mutexWork = Lock()
        self.__work = 0
        self.__full = Semaphore(0)

        # wait for workers to connect
        start_new_thread(self.__workerHandler, ())

    def close(self):
        self.__mainSocket.close()

    def __threaded_client(self, workerID, taskID, rows, cols, subResults, offloadAttempt=0):
        worker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # try to establish connection to worker
        try:
            worker.connect(workerID)

            rowsShapeStr = str(rows.shape)
            colsShapeStr = str(cols.shape)
            taskHeader = str(taskID) + '|' + rowsShapeStr + '|' + colsShapeStr
            taskHeaderConf = str(taskID) + '=' + rowsShapeStr + 'x' + colsShapeStr

            worker.send(str.encode(taskHeader))

            taskHeaderResponseRaw = worker.recv(DEF_HEADER_SIZE)
            if(taskHeaderResponseRaw == None):
                raise ConnectionAbortedError('worker failed @ task header response')

            taskHeaderResponse = taskHeaderResponseRaw.decode('utf-8')
            if(taskHeaderConf != taskHeaderResponse):
                raise ValueError('Task not received correctly')

            # send the first part of matrix
            send_mm(worker, rows)
           # check if the connection is still alive
            matARecRaw = worker.recv(DEF_HEADER_SIZE)
            # check if the connection is still alive
            if(matARecRaw == None):
                raise ConnectionAbortedError('worker failed @ matrix a confirmation')
            # decode the message and check it corresponds with what was sent
            matARec = matARecRaw.decode('utf-8')
            if(matARec != rowsShapeStr):
                raise ValueError('Matrix A = ' + matARec)

            # send the second part of matrix
            send_mm(worker, cols)
            # check if the connection is still alive
            matBRecRaw = worker.recv(DEF_HEADER_SIZE)
            # check if the connection is still alive
            if(matBRecRaw == None):
                raise ConnectionAbortedError('worker failed @ matrix b confirmation')
            # decode the message and check it corresponds with what was sent
            matBRec = matBRecRaw.decode('utf-8')
            if(matBRec != colsShapeStr):
                raise ValueError('Matrix B = ' + matBRec)

            # send message to inform worker the main node is ready to receive results
            worker.send(str.encode('start work!'))

            # save the results to the appropriate position in the subResults matrix
            # using the taskID which indicates the row and column index
            results = recv_mm(worker)
            if(type(results) != np.ndarray):
                raise ConnectionAbortedError("No matrix results received")
            # save the result
            subResults[taskID[0]][taskID[1]] = results

            # place the worker back into the free set of the worker list
            self.__mutex.acquire()
            self.__wList.freeWorker(workerID)
            self.__mutex.release()
            # close the worker connection
            worker.close()

        except Exception as e:
            worker.close()
            if offloadAttempt == self.__offloadAttempts:
                print(str(e))
                raise e
            self.__waitForWorker(1, 3, 1)
            # get worker
            altWorkerID = self.__wList.getWorker()
            self.__threaded_client(altWorkerID, taskID, rows, cols, subResults, offloadAttempt + 1)

    def __workerHandler(self):
        while True:
            # wait for new workers to join
            worker, address = self.__mainSocket.accept()
            # send the port back to the worker on which the connection will be made
            worker.send(str.encode(str(address[1])))
            # when a worker add the worker to '__wList'
            self.__mutex.acquire()  # acquire lock for '__wList' variable
            self.__wList.addWorker(address)  # save the worker connection info
            self.__mutex.release()  # release lock for '__wList'  variable
            # close the initial connection
            worker.close()

    def __waitForWorker(self, sleepTime, numTries, reqWorkers):
        tries = 0
        while True:
            # waited (__timeOut * __tries) seconds with no success, then raise exception
            if tries > numTries:
                self.__mainSocket.close()
                raise Exception('Minimum number workers (' + str(self.__taskSplit + 1) + ') not available.')
            self.__mutex.acquire()  # acquire lock for '__wList' variable

            if self.__wList.freeSize() >= reqWorkers:
                # if there are enough workers then break out of this loop
                self.__mutex.release()  # release lock for '__wList'  variable
                break
            elif (self.__wList.occupiedSize() + self.__wList.freeSize()) > reqWorkers:
                # if there are enough workers but they are busy then reset the tries because
                # the task can be distributed at some point in the future
                tries = 0

            self.__mutex.release()  # release lock for '__wList'  variable
            # if there are not enough workers then sleep for a couple of seconds
            tries += 1
            time.sleep(sleepTime)

    def distributeWork(self, mat_a, mat_b):
        if (mat_a.shape[0] != mat_b.shape[1]) or (mat_a.shape[1] != mat_b.shape[0]):
            return None

        # check enough workers are available to offload the task
        try:
            self.__waitForWorker(self.__sleepTime, self.__tries, self.__taskSplit + 1)
        except Exception as e:
            raise e

        # acquire the mutex to get access to the worker list
        self.__mutex.acquire()

        # squre root of the numbers of tasks to evenly split tasks
        SSplit = int(sqrt(self.__taskSplit))

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

                # assign random id to task which will be used during communication
                subTaskID = (i, j)
                # get worker
                worker = self.__wList.getWorker()
                # start a worker thread
                wt = Thread(name=str(subTaskID), target=self.__threaded_client, args=(worker, subTaskID, mat_a[sR:eR, :], mat_b[:, sC:eC], subResults))
                wt.start()
                wThreads.append(wt)

        # release the mutex to free the worker list
        self.__mutex.release()

        # wait for all worker threads to finish then
        for i in range(len(wThreads)):
            wThreads[i].join()

        return np.bmat(subResults)
