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
# define size of matrix
r = 12000
c = 12000
rc = r * c
# create two matrices of
# create two matrices of of the given dimension
mat_a = (np.random.randint(100, size=(r, c))).astype(np.float64)
mat_b = (np.random.randint(100, size=(c, r))).astype(np.float64)

def baseline():
    print('\nLocal Compute test')
    
    print('Computing multiplication of', (r, c), 'x', (c, r))

    tStart1 = time.time_ns()
    sRes = np.matmul(mat_a, mat_b)
    tEnd1 = time.time_ns()

    cTime1 = (tEnd1 - tStart1) / 1000000000.0
    print('\nLocal Compute Time:', round(cTime1, 2), 's')

def split4():
    dd.taskSplit = 4
    print('\nSmall granularity: split task into 4 subtasks')
    
    print('\nComputing multiplication of', (r, c), 'x', (c, r))

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
    print('\n4 sub-tasks compute time:', round(cTime1, 2), 's')

def split6():
    dd.taskSplit = 6
    print('\nSmall granularity: split task into 6 subtasks')
    
    print('\nComputing multiplication of', (r, c), 'x', (c, r))

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
    print('\n6 sub-tasks compute time:', round(cTime1, 2), 's')

def split8():
    dd.taskSplit = 8
    print('\nSmall granularity: split task into 8 subtasks')
    
    print('\nComputing multiplication of', (r, c), 'x', (c, r))

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
    print('\n8 sub-tasks compute time:', round(cTime1, 2), 's')

def split9():
    dd.taskSplit = 9
    print('\nSmall granularity: split task into 9 subtasks')
    
    print('\nComputing multiplication of', (r, c), 'x', (c, r))

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
    print('\n9 sub-tasks compute time:', round(cTime1, 2), 's')

def main():
    try:
        print('\n###############################################################################\n')
        print('                                  Granularity Test                                 ')
        print('\n###############################################################################\n')
        _ = input()

        # baseline
        print('\n###############################################################################\n')
        baseline()
        sleep(1)
        
        # 4
        print('\n###############################################################################\n')
        split4()
        sleep(1)
        print('\n###############################################################################\n')
        split4()
        sleep(1)
        print('\n###############################################################################\n')
        split4()
        sleep(1)

        # 6
        print('\n###############################################################################\n')
        split6()
        sleep(1)
        print('\n###############################################################################\n')
        split6()
        sleep(1)
        print('\n###############################################################################\n')
        split6()
        sleep(1)

        # 8
        print('\n###############################################################################\n')
        split8()
        sleep(1)
        print('\n###############################################################################\n')
        split8()
        sleep(1)
        print('\n###############################################################################\n')
        split8()
        sleep(1)

        # 9
        print('\n###############################################################################\n')
        split9()
        sleep(1)
        print('\n###############################################################################\n')
        split9()
        sleep(1)
        print('\n###############################################################################\n')
        split9()

    except KeyboardInterrupt as e:
        print('')
        # shut down the main node
        dd.close()


if __name__ == "__main__":
    # execute only if run as a script
    main()
