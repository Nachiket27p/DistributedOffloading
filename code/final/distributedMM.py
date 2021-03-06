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
    def __init__(self, hostIP='127.0.0.1', port=5000, maxListenQ=10, taskSplit=4, tries=10, sleepTime=2.0, offloadAttempts=1, spareWorkers=1, caching=True) -> None:
        """
        Construct an object capable of performing distributed matrix multiplication.
        Useses socket programming to listen for connections from worker nodes,
        onece worker nodes have connected their connection info is saved. This info is
        used when a distributed matrix multiplication is requested to offload work to
        the worker nodes.

        Args:
            hostIP (str, optional): The IP address of the host/main node.
                                    Defaults to '127.0.0.1'.
            port (int, optional): The port on which to accept worker connections.
                                    Defaults to 5000.
            maxListenQ (int, optional): The number of requests which can be in the queue at the same time.
                                            Defaults to 10.
            taskSplit (int, optional): The square root of the number of task to split into.
                                        Defaults to 2.
            tries (int, optional): Used to compute the max amount of wait time using the formula, tries*sleepTime.
                                    Defaults to 5.
            sleepTime (float, optional): The amount of time to sleep each after attempt to obtain workers for a task.
                                            Defaults to 2.0.
            offloadAttempts (int, optional): The number of times to re-offload a task when a worker fails.
                                                Defaults to 1.
            spareWorkers (int, optional): Number of spare workers to in addition to the required amount to distribute the task.
                                            Defaults to 1.

        Raises:
            Exception: If a socket cannot be opened on the specified IP:Port.
        """
        self.__hostIP = hostIP
        self.__port = port
        self.__mainSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__maxListenQ = maxListenQ
        # ensure the task split is always even
        if (taskSplit < 4):
            self.__taskSplit = 4
        elif ((taskSplit % 2) == 1):
            self.__taskSplit = taskSplit - 1
        else:
            self.__taskSplit = taskSplit
        self.__tries = tries
        self.__sleepTime = sleepTime
        self.__offloadAttempts = offloadAttempts
        self.__spareWorkers = spareWorkers
        self.__caching = caching

        try:
            self.__mainSocket.bind((self.__hostIP, self.__port))
        except socket.error as e:
            raise Exception(str(e))

        self.__mainSocket.listen(self.__maxListenQ)

        # worker list
        self.__wList = WorkerList()
        # locks to protect the worker list
        self.__mutex = Lock()
        self.__work = 0
        self.__full = Semaphore(0)

        # wait for workers to connect
        start_new_thread(self.__workerHandler, ())

        # splittting map
        self.__maxSplitMap = 16
        self.__splitMap = {4:(2,2), 6:(3,2), 8:(4,2), 10:(3,3), 12:(4,3), 14:(4,4), 16:(4,4)}

    @property
    def hostIP(self):
        """Get the host IP

        Returns:
            string: Host IP in string form
        """
        return self.__hostIP

    @property
    def port(self):
        """Get the port number on which the server will accept new workers.

        Returns:
            int: Port number
        """
        return self.__port
    
    @property
    def maxListenQ(self):
        """Get the maximum number of connection requests which can be queued up in terms of 
        the number of workers trying to connect.

        Returns:
            int: Max connection requests which can be queued up.
        """
        return self.__maxListenQ

    @property
    def taskSplit(self):
        """Get the number of sub-tasks the next matmul call split the work into.

        Returns:
            int: Number of sub-tasks
        """
        return self.__taskSplit
    
    @property
    def tries(self):
        """Get the numbe of times the main thread will try to check if there are enough workers
        to offload the multiplication requested by matmul.

        Returns:
            int: Number of tries.
        """
        return self.__tries
    
    @property
    def sleepTime(self):
        """Amount of times the connection handler thread will sleep after trying to accept
        connection from worker nodes.

        Returns:
            float: Amount of time to sleep in seconds.
        """
        return self.__sleepTime
    
    @property
    def offloadAttempts(self):
        """Get the number of times a specific sub-task can fail and be re-offloaded before
        giving up and the matmul function will throw an exception.

        Returns:
            int: Number of re-offloads of a task under worker node failure.
        """
        return self.__offloadAttempts
    
    @property
    def spareWorkers(self):
        """Get the number of spare workers to keep in the event of node failues.

        Returns:
            int: Number of workers not being utilized for distributed matrix multiplication
        """
        return self.__spareWorkers
    
    @property
    def caching(self):
        """Get the current state of the caching mechanism, if the value is Trie then
        the system is performing caching of the compressed data which gets sent to
        worker node. If False, then the system is not caching.

        Returns:
            Bool: True if cahcing is enabled, False if it is disabled
        """
        return self.__caching
    
    @taskSplit.setter
    def taskSplit(self, value):
        """Used to set the number of subtasks the requested matrix multiplication should be
        split up into. The value provided should be 'even'. If an 'odd' value is provided
        system will default to 'value-1'. If the value is less than 4 then the system will
        default to 4.

        Args:
            value (int): Number of sub-tasks the matrix multiplication should be split into.
        """
        if (value < 4):
            self.__taskSplit = 4
        elif ((value % 2) == 1):
            self.__taskSplit = value - 1
        else:
            self.__taskSplit = value
    
    @tries.setter
    def tries(self, value):
        """Set the number of times the matmul function call should try and obtain enough
        workers to distribute the workload.

        Args:
            value (int): Number of times to retry and get the required workers.
        """
        if(value < 1):
            self.__tries = 1
        else:
            self.__tries = value

    @sleepTime.setter
    def sleepTime(self, value):
        """Set the amount of time in seconds the new worker connection thread should wait
        before re-checking for new worker connection requests.

        Args:
            value (float): Amount of time to sleep before retrying to connecto more workers
        """
        if(value < 1):
            self.__sleepTime = 1.0
        else:    
            self.__sleepTime = value
    
    @spareWorkers.setter
    def spareWorkers(self, value):
        """Set the number of spare workers to keep. If the value provided is less than 1 then
        the value will default to 1.

        Args:
            value (int): The number of spare workers.
        """
        if (value < 1):
            self.__spareWorkers = 1
        else:    
            self.__spareWorkers = value

    @offloadAttempts.setter
    def offloadAttempts(self, value):
        """Set the number of re-offload attempts under the situation where a worker node
        fails/loses connection.

        Args:
            value (int): Number of re-offload attempts.
        """
        if(value < 1):
            self.__offloadAttempts = 1
        else:
            self.__offloadAttempts = value
    
    @caching.setter
    def caching(self, value):
        """USed to enable or disable caching. Accepts a boolean value to enable or disable
        the compressed data caching mechanism.

        Args:
            value (Bool): True if you want to enable caching, False if you want to disable caching.
        """
        self.__caching = value

    def close(self):
        """
            Closes the main server socket listener
        """
        self.__mainSocket.close()

    def __threaded_client(self, workerID, taskID, rows, cols, subResults, compDataCache, atol=None, offloadAttempt=0):
        """
        Private function which is executed with independent threads for each worker.
        This function can is recursively called in the event of a worker node failure.
        The number of times this the recursion (# number of re-offloads) can be specified when
        constructing the class.

        Args:
            workerID (tuple): With two elements, which allow the thread to index into subResults to save the results.
            taskID (tuple): With three elements:
            first: 
            rows (ndarray): The segment of the first matrix
            cols (ndarray): The segment of the second matrix
            subResults (array): An array which keeps track of the results
            offloadAttempt (int, optional): Keeps tack of this re-offload attempt. Defaults to 0.
            compDataCache (dict): A dictionary which keeps track of the compressed data to ensure
                                    in the event of a re-offload, the data does not need to be
                                    compressed again.

        Raises:
            ConnectionAbortedError: Internally raied if the connection is severed
            ValueError: Internally raised if the worker responded with an invalid response
            e: Raised if the number of re-offloads is exceeded, this exception can be either
                be a ConnectionAbortedError or ValueError.
        """
        worker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # try to establish connection to worker
        try:
            worker.connect(workerID)

            rowsShapeStr = str(rows.shape)
            colsShapeStr = str(cols.shape)
            taskHeader = str(taskID) + '|' + rowsShapeStr + '|' + colsShapeStr + '|' + str(atol)
            taskHeaderConf = str(taskID) + '=' + rowsShapeStr + 'x' + colsShapeStr

            worker.send(str.encode(taskHeader))

            taskHeaderResponseRaw = worker.recv(DEF_HEADER_SIZE)
            if(taskHeaderResponseRaw == None):
                raise Exception('worker failed @ task header response')

            taskHeaderResponse = taskHeaderResponseRaw.decode('utf-8')
            if(taskHeaderConf != taskHeaderResponse):
                raise Exception('Task not received correctly')

            # send the first part of matrix, also considering if the caching mechanism is enabled
            if (self.__caching):
                send_mm(worker, rows, atol, compDataCache, taskID[1])
            else:
                send_mm(worker, rows, atol, compDataCache, None)
            # check if the connection is still alive
            matARecRaw = worker.recv(DEF_HEADER_SIZE)
            # check if the connection is still alive
            if(matARecRaw == None):
                raise Exception('worker failed @ matrix a confirmation')
            # decode the message and check it corresponds with what was sent
            matARec = matARecRaw.decode('utf-8')
            if(matARec != rowsShapeStr):
                raise Exception('Matrix A = ' + matARec)

            # send the second part of matrix, also considering if the caching mechanism is enabled
            if (self.__caching):
                send_mm(worker, cols, atol, compDataCache, taskID[2])
            else:
                send_mm(worker, cols, atol, compDataCache, None)
            # check if the connection is still alive
            matBRecRaw = worker.recv(DEF_HEADER_SIZE)
            # check if the connection is still alive
            if(matBRecRaw == None):
                raise Exception('worker failed @ matrix b confirmation')
            # decode the message and check it corresponds with what was sent
            matBRec = matBRecRaw.decode('utf-8')
            if(matBRec != colsShapeStr):
                raise Exception('Matrix B = ' + matBRec)

            # send message to inform worker the main node is ready to receive results
            worker.send(str.encode('start work!'))

            # save the results to the appropriate position in the subResults matrix
            # using the taskID which indicates the row and column index
            results = recv_mm(worker)
            if(type(results) != np.ndarray):
                raise Exception("No matrix results received")
            # save the result
            subResults[taskID[0][0]][taskID[0][1]] = results

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
            self.__threaded_client(altWorkerID, taskID, rows, cols, subResults, compDataCache, atol, offloadAttempt + 1)

    def __workerHandler(self):
        """
        Private method which executed on an asynchronous thread, which establishes connections with
        worker nodes and saves their connection info into a worker list.
        """
        while True:
            # wait for new workers to join
            worker, address = self.__mainSocket.accept()
            # print('Worker', address, 'connected')
            # send the port back to the worker on which the connection will be made
            worker.send(str.encode(str(address[1])))
            # when a worker add the worker to '__wList'
            self.__mutex.acquire()  # acquire lock for '__wList' variable
            self.__wList.addWorker(address)  # save the worker connection info
            self.__mutex.release()  # release lock for '__wList'  variable
            # close the initial connection
            worker.close()

    def __waitForWorker(self, sleepTime, numTries, reqWorkers, spareWorkers=0):
        """
        A private which is used to make a request to obtain a spcific number of workers.
        The request procedure uses a mutex variable to protect the shared worker list.
        Three parameter have to be specified, the anount of time to sleep if enough workers
        are not available, the number of time to try acuiring workers, and the number of 
        workers being requested.

        Args:
            sleepTime (flot): The amount of time to sleep
            numTries (int): Number of times to try and get the requested workers
            reqWorkers (int): Number of workers being requested
            spareWorkers (int): Number of spare workers to keep, can be used to mitigate the overhead of node failures.
                                Defaults to 0.

        Raises:
            Exception: If enough workers cannot be obtained within the given constraints
                        throw an exception to indicate the task can no longer be accomplished.
        """
        tries = 0
        while True:
            # waited (__timeOut * __tries) seconds with no success, then raise exception
            if tries > numTries:
                self.__mainSocket.close()
                raise Exception('Minimum number workers (' + str(self.__taskSplit + self.__spareWorkers) + ') not available.')

            # acquire lock for '__wList' variable
            self.__mutex.acquire()

            if (self.__wList.freeSize() >= (reqWorkers + spareWorkers)):
                # if there are enough workers then break out of this loop
                self.__mutex.release()  # release lock for '__wList'  variable
                break
            elif ((self.__wList.occupiedSize() + self.__wList.freeSize()) >= (reqWorkers + spareWorkers)):
                # if there are enough workers but they are busy then reset the tries because
                # the task can be distributed at some point in the future
                tries = 0

            self.__mutex.release()  # release lock for '__wList'  variable
            # if there are not enough workers then sleep for a couple of seconds
            tries += 1
            time.sleep(sleepTime)

    def matmul(self, mat_a, mat_b, atol=None):
        """
        This method can be called to distribute the multiplication of the two matrices provided.
        The two matrices have to be numpy arrays, for this operation to work. This operation
        works for floating point and integer values.
        Note that for integer multilpication numpy does not automatically harness multithreading
        so the advantage gained will seem much larger than when using floating point values because,
        floating point multiplication is automatically multithreaded by numpy.
        Under the hood the multiplication is done using numpy.matmul().

        Args:
            mat_a (ndarray): The first of the two matrices being multiplied
            mat_b (ndarray): The second of the two matrices being multiplied

        Raises:
            e: If the numbe of workers required to distribute the workload are not available
                within the constrains defined by:
                    self.__sleepTime, self.__tries, self.__taskSplit
                Which are defined at the beginning of 

        Returns:
            ndarray: A numpy matrix representing the result of the matrix multiplication
        """
        if (mat_a.shape[0] != mat_b.shape[1]) or (mat_a.shape[1] != mat_b.shape[0]):
            return None

        # check enough workers are available to offload the task
        try:
            self.__waitForWorker(self.__sleepTime, self.__tries, self.__taskSplit, self.__spareWorkers)
        except Exception as e:
            raise e

        # acquire the mutex to get access to the worker list
        self.__mutex.acquire()

        rSplit = cSplit = None
        # if the task split specified is greater than 16 then always find the square root
        # to determine the task split amount
        if (self.__taskSplit > self.__maxSplitMap):
            rSplit = cSplit = int(sqrt(self.__taskSplit))
        else:
            # get the task split for the split specified from the map
            tSplit = self.__splitMap[self.__taskSplit]
            # if the number of rows is greater than the columns
            # then row wise with larger split value (first of the tuple returned from map)
            if mat_a.shape[0] > mat_a.shape[1]:
                rSplit = tSplit[0]
                cSplit = tSplit[1]
            else:
                rSplit = tSplit[1]
                cSplit = tSplit[0]
            
        subResults = [[None] * cSplit for _ in range(rSplit)]

        wThreads = []
        compDataCache = dict()

        for i in range(rSplit):
            # compute the start and end row for the sub task for matrix a
            sR = (mat_a.shape[0] // rSplit) * i
            # if this is the one of the corner/end segments
            # make the end index -1 to indicate the last one
            eR = sR + (mat_a.shape[0] // rSplit)
            if (i == (rSplit - 1)):
                eR = mat_a.shape[0]

            for j in range(cSplit):
                # compute the start and end column for the sub task for matrix b
                sC = (mat_b.shape[1] // cSplit) * j
                # if this is the one of the corner/end segments
                # make the end index -1 to indicate the last one
                eC = sC + (mat_b.shape[1] // cSplit)
                if (j == (cSplit - 1)):
                    eC = mat_b.shape[1]

                # assign random id to task which will be used during communication
                subTaskID = ((i, j), ('r', sR, eR), ('c', sC, eC))

                # get worker
                worker = self.__wList.getWorker()
                # start a worker thread
                wt = None
                if (self.__caching):
                    wt = Thread(name=str(subTaskID), target=self.__threaded_client, args=(worker, subTaskID, mat_a[sR:eR, :], mat_b[:, sC:eC], subResults, compDataCache, atol, 0))
                else:
                    wt = Thread(name=str(subTaskID), target=self.__threaded_client, args=(worker, subTaskID, mat_a[sR:eR, :], mat_b[:, sC:eC], subResults, None, atol, 0))
                wt.start()
                wThreads.append(wt)

        # release the mutex to free the worker list
        self.__mutex.release()

        # wait for all worker threads to finish then
        for i in range(len(wThreads)):
            wThreads[i].join()

        return np.bmat(subResults)
