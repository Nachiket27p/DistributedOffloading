import numpy as np
from math import sqrt
import time
from time import sleep
import socket
from distributedMM import DMM
from os import system

# condigure address for socket
ip = socket.gethostbyname(socket.gethostname())
# ip = '127.0.0.1'
p = 5005
# object to access the distributed offloading mechanism
dd = DMM(hostIP=ip, port=p, taskSplit=4)


def testCorrectness():
    print('Testing correctness of the result obtained from distributed matrix multiplication.')
    # define size of matrix
    r = 8
    c = 8
    rc = r * c

    # create two matrices of of the given dimension
    mat_a = (np.random.randint(100, size=(r, c))).astype(np.float64)
    mat_b = (np.random.randint(100, size=(c, r))).astype(np.float64)

    print('Computing multiplication of', (r, c), 'x', (c, r))

    # compute results on local machine
    sRes = np.matmul(mat_a, mat_b)

    try:
        dRes = dd.matmul(mat_a, mat_b)
    except Exception as e:
        print(str(e))
        dd.close()
        exit()

    print('\nResult from computing on single machine.')
    print(sRes)

    # perform distributed computation
    print('\nResults from distributed computation.')
    print(dRes)

    print('\nDifference:')
    print(sRes - dRes)

    # check if the results match
    print('\nResults Match:', np.all(sRes == dRes))

    # wait for input
    _ = input()


def testComputeTimeAdvantage():
    print('\nTesting compute time advantage of distributed matrix multiplication.')
    # define size of matrix
    r = 10000
    c = 10000
    rc = r * c
    # create two matrices of
    # create two matrices of of the given dimension
    mat_a = (np.random.randint(100, size=(r, c))).astype(np.float64)
    mat_b = (np.random.randint(100, size=(c, r))).astype(np.float64)

    print('Computing multiplication of', (r, c), 'x', (c, r))

    tStart1 = time.time_ns()
    sRes = np.matmul(mat_a, mat_b)
    tEnd1 = time.time_ns()

    cTime1 = (tEnd1 - tStart1) / 1000000000.0
    print('\nLocal Compute Time:', round(cTime1, 2), 's')

    tStart2 = time.time_ns()
    try:
        dRes = dd.matmul(mat_a, mat_b)
    except Exception as e:
        print(str(e))
        dd.close()
        exit()
    tEnd2 = time.time_ns()

    # report the time taken to compute multiplication through distributed method
    cTime2 = (tEnd2 - tStart2) / 1000000000.0
    print('\nDistributed Compute Time:', round(cTime2, 2), 's')
    print('\nTime saved:', round(cTime1 - cTime2, 2), 's')
    # wait for input
    _ = input()


def testNodeFailure():
    print('Testing the ability to handle a node failure and showing the overhead of node failure.')
    # define size of matrix
    r = 10000
    c = 10000
    rc = r * c
    # create two matrices of
    # create two matrices of of the given dimension
    mat_a = (np.random.randint(100, size=(r, c))).astype(np.float64)
    mat_b = (np.random.randint(100, size=(c, r))).astype(np.float64)

    print('Computing multiplication of', (r, c), 'x', (c, r))

    tStart1 = time.time_ns()
    try:
        dRes = dd.matmul(mat_a, mat_b)
    except Exception as e:
        print(str(e))
        dd.close()
        exit()
    tEnd1 = time.time_ns()

    cTime1 = (tEnd1 - tStart1) / 1000000000.0
    # report the time taken to compute multiplication through distributed method
    print('\nDistributed compute time without node failure:', round(cTime1, 2), 's')

    tStart2 = time.time_ns()
    try:
        dRes = dd.matmul(mat_a, mat_b)
    except Exception as e:
        print(str(e))
        dd.close()
        exit()
    tEnd2 = time.time_ns()

    cTime2 = (tEnd2 - tStart2) / 1000000000.0
    # report the time taken to compute multiplication through distributed method
    print('\nDistributed Compute Time with 1 node failure:', round(cTime2, 2), 's')

    print('\nOverhead of 1 node failure:', round(cTime2 - cTime1, 2), 's')

    # wait for input
    _ = input()


def main():
    try:
        while True:
            system('clear')
            testNumber = input("\nTest case: ")
            system('clear')
            print('\n###############################################################################\n')
            print('                                  Test case:', testNumber)
            print('\n###############################################################################\n')
            if(testNumber == '1'):
                testCorrectness()
            elif(testNumber == '2'):
                testComputeTimeAdvantage()
            elif(testNumber == '3'):
                testNodeFailure()
    except KeyboardInterrupt as e:
        print('')
        # shut down the main node
        dd.close()


if __name__ == "__main__":
    # execute only if run as a script
    main()
