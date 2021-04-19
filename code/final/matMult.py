from sys import call_tracing
from distributedMM import DMM
import numpy as np
from math import sqrt
import time
from time import sleep

r = 512
c = 512
rc = r * c
mat_a = (np.arange(rc).reshape(r, c)).astype(np.float)
mat_b = (np.arange(rc).reshape(r, c)).astype(np.float)

tStart = time.time_ns()
singRes = np.matmul(mat_a, mat_b)
tEnd = time.time_ns()
print('Local Compute Time:', float((tEnd - tStart)) / 100000000.0)


dd = DMM(port=5001)

# workers to connect
sleep(5)

tStart = time.time_ns()
try:
    dRes1 = dd.distributeWork(mat_a, mat_b)
except Exception as e:
    print(str(e))
    exit()
tEnd = time.time_ns()

print('Distributed Compute Time:', float((tEnd - tStart)) / 100000000.0)

print('Results Match:', np.all(singRes == dRes1))

# sleep(2)

# try:
#     dRes2 = dd.distributeWork(mat_a, mat_b)
#     print('----------------------------------------------')
#     print(dRes2)
# except Exception as e:
#     print(str(e))
#     exit()


dd.close()

# split = 4
# SSplit = int(sqrt(split))
# # print(SSplit)

# res = [[None] * SSplit for _ in range(SSplit)]

# for i in range(SSplit):
#     sR = (len(mat_a) // SSplit) * i
#     eR = sR + (len(mat_a) // SSplit)
#     for j in range(SSplit):
#         sC = (len(mat_b[0]) // SSplit) * j
#         eC = sC + (len(mat_b[0]) // SSplit)

#         # print(sR, eR, sC, eC)
#         # start a worker thread
#         res[i][j] = np.matmul(mat_a[sR:eR, :], mat_b[:, sC:eC])

# distRes = np.bmat(res)
# print(distRes)
