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

    def __init__(self, hostIP='127.0.0.1', port=5000, maxListenQ=5) -> None:
        self.__hostIP = hostIP
        self.__port = port
        self.__mainSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__maxListenQ = maxListenQ
        self.__taskSplit = 4
        self.__tries = 5
        self.__sleepTime = 2.0

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

    def __threaded_client(self, workerID, taskID, rows, cols, subResults):
        worker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # try to establish connection to worker
        try:
            worker.connect(workerID)
        except socket.error as e:
            worker.close()
            print('connection to ' + str(workerID) + ' failed with: ' + str(e))
            return

        rowsShapeStr = str(rows.shape)
        colsShapeStr = str(cols.shape)
        taskHeader = str(taskID) + '|' + rowsShapeStr + '|' + colsShapeStr
        taskHeaderConf = str(taskID) + '=' + rowsShapeStr + 'x' + colsShapeStr

        worker.send(str.encode(taskHeader))

        taskHeaderResponse = (worker.recv(DEF_HEADER_SIZE)).decode('utf-8')

        if(taskHeaderConf == taskHeaderResponse):
            send_mm(worker, rows)
            mat_a_raw = worker.recv(DEF_HEADER_SIZE)
            print(mat_a_raw)
            mat_a_rec = mat_a_raw.decode('utf-8')
            if(mat_a_rec != rowsShapeStr):
                print(mat_a_rec)
                return

            send_mm(worker, cols)
            mat_b_raw = worker.recv(DEF_HEADER_SIZE)
            print(mat_b_raw)
            mat_b_rec = mat_b_raw.decode('utf-8')
            if(mat_b_rec != colsShapeStr):
                print(mat_b_rec)
                return
        else:
            print('Task header error')
            return

        # send message to inform ready to recv
        worker.send(str.encode("send it!"))

        # save the results to the appropriate position in the subResults matrix
        # using the taskID which indicates the row and column index
        subResults[taskID[0]][taskID[1]] = recv_mm(worker)

        # place the worker back into the free set of the worker list
        self.__mutex.acquire()
        self.__wList.freeWorker(workerID)
        self.__mutex.release()
        # close the worker connection
        worker.close()

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

    def distributeWork(self, mat_a, mat_b):
        if (mat_a.shape[0] != mat_b.shape[1]) or (mat_a.shape[1] != mat_b.shape[0]):
            return None

        tries = 0
        while True:
            # waited (__sleepTime * __tries) seconds with no success, then raise exception
            if tries > self.__tries:
                raise Exception('Minimum number workers (' + str(self.__taskSplit + 1) + ') not available.')
            self.__mutex.acquire()  # acquire lock for '__wList' variable

            if self.__wList.freeSize() >= (self.__taskSplit):
                # if there are enough workers then break out of this loop
                self.__mutex.release()  # release lock for '__wList'  variable
                break

            self.__mutex.release()  # release lock for '__wList'  variable
            # if there are not enough workers then sleep for a couple of seconds
            tries += 1
            time.sleep(self.__sleepTime)

        #!
        # time.sleep(self.__sleepTime)
        #!

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

                # get worker
                worker = self.__wList.getWorker()
                # assign random id to task which will be used during communication
                subTaskID = (i, j)
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
