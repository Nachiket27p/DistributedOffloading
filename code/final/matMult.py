from sys import call_tracing
from distributedMM import DMM
import numpy as np
from math import sqrt
import time

r = 128
c = 128
rc = r * c
mat_a = (np.arange(rc).reshape(r, c)).astype(np.float)
mat_b = (np.arange(rc).reshape(r, c)).astype(np.float)

singRes = np.matmul(mat_a, mat_b)
print(singRes)

dd = DMM(port=5000)

dRes = dd.distributeWork(mat_a, mat_b)
print(dRes)

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
