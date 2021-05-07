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
dd = DMM(hostIP=ip, port=p)
dd.taskSplit = 6

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

    # # compute results on local machine
    # sRes = np.matmul(mat_a, mat_b)

    print('Computing multiplication of', (r, c), 'x', (c, r))

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
    # print(np.array_equal(sRes, dRes))

def main():
    try:
        print('\n###############################################################################\n')
        print('                                  Node Failures                                 ')
        print('\n###############################################################################\n')
        _ = input()

        testNodeFailure()

    except KeyboardInterrupt as e:
        print('')
        # shut down the main node
        dd.close()
    
    dd.close()


if __name__ == "__main__":
    # execute only if run as a script
    main()
