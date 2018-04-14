import pprint
from math import sqrt
import numpy as np
import scipy
import scipy.linalg  # SciPy Linear Algebra Library

# A = np.array([[1., 2., 3.], [2., 5., 7.], [3., 5., 3.]])
#A = np.array([[4, 3], [6, 3]])
A = np.array([[6, 1, 4],
              [4, 8, 4],
              [6, 3, 5]])
n = 3

# 初始化L、U
L = [[0 for _ in range(n)] for _ in range(n)]
U = [[0 for _ in range(n)] for _ in range(n)]

for i in range(n):
    U[0][i] = A[0][i]

for i in range(n):
    L[i][i] = 1
    L[i][0] = A[i][0] / U[0][0]
# 纵向迭代
for t in range(n):
    print("第%s次迭代\n" % t)
    for i in range(t, n):
        sum_LU = 0
        for k in range(t):
            sum_LU += L[t][k] * U[k][i]
        U[t][i] = A[t][i] - sum_LU
        sum_L = 0
        for k in range(t):
            sum_L += L[i][k] * U[k][t]
        L[i][t] = (A[i][t] - sum_L) / U[t][t]
    print("U矩阵:\n", np.array(U))
    print("L矩阵:\n", np.array(L), "\n")


print("A:")
print(A)

print("L:")
print(np.array(L))

print("U:")
print(np.array(U))

# 横向迭代
# for t in range(n):
#     for i in range(t + 1, n):
#         sum_LU = L[t + 1][t] * U[t][t + 1]
#         for k in range(t):
#             sum_LU += L[t + 1][k] * U[k][t + 1]
#         # sum_LU = 0
#         # for k in range(t + 1):
#         #     sum_LU += L[t + 1][k] * U[k][t + 1]
#         U[t + 1][i] = A[t + 1][i] - sum_LU
"""
# 进入迭代
# 第一次纵向迭代
for i in range(1, n):
    L[i][0] = A[i][0] / U[0][0]
# 第一次横向迭代
for i in range(1, n):
    U[1][i] = A[1][i] - L[1][0] * U[0][1]

# 第二次纵向迭代
for i in range(2, n):
    L[i][1] = (A[i][1] - L[i][1]) / U[1][1]
# 第二次横向迭代
for i in range(2, n):
    U[2][i] = A[2][i] - L[2][0] * U[0][2] - L[2][1] * U[1][2]

# 第三次纵向迭代
for i in range(3, n):
    L[i][2] = (A[i][2] - L[i][1] - L[i][2]) / U[2][2]

# 第三次横向迭代
"""
