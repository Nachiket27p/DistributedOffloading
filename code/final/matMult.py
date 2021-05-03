from sys import call_tracing
from distributedMM import DMM
import numpy as np
from math import sqrt
import time
from time import sleep
import socket

# condigure address for socket
ip = socket.gethostbyname(socket.gethostname())
# ip = '127.0.0.1'
p = 5005
# object to access the distributed offloading mechanism
dd = DMM(hostIP=ip, port=p)
# set task split size
dd.taskSplit = 6

# wait for workers to connect
print("Waiting for workers")
sleep(5)
print("Starting work")

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

print(np.array_equal(sRes, dRes))

# close the distributed server
dd.close()
