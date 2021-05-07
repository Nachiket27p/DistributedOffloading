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

def splitTask(r, c):
    rc = r * c
    # create two matrices of
    # create two matrices of of the given dimension
    mat_a = (np.random.randint(100, size=(r, c))).astype(np.float64)
    mat_b = (np.random.randint(100, size=(c, r))).astype(np.float64)
    
    print('\n6 way split of', (r, c), 'x', (c, r))

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
    print('\nCompute time:', round(cTime1, 2), 's')

def localCompute(r, c):
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

def main():
    try:
        print('\n###############################################################################\n')
        print('                                  Size of Matrix                                 ')
        print('\n###############################################################################\n')
        _ = input()

        # n = 2
        # while(n <= 16):
        #     r = c = (1000*n)
        #     print('\n###############################################################################\n')
        #     localCompute(r, c)
        #     sleep(1)
        #     n = n + 2
        
        n = 1
        while(n <= 16):
            r = c = (1000*n)
            print('\n###############################################################################\n')
            localCompute(r, c)
            sleep(1)
            print('\n###############################################################################\n')
            splitTask(r, c)
            sleep(1)
            print('\n###############################################################################\n')
            splitTask(r, c)
            sleep(1)
            print('\n###############################################################################\n')
            splitTask(r, c)
            sleep(1)
            n = n*2


    except KeyboardInterrupt as e:
        print('')
        # shut down the main node
        dd.close()


if __name__ == "__main__":
    # execute only if run as a script
    main()
