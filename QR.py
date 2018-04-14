# coding: utf-8

import numpy as np
from numpy.random import randint
import math
from numpy.linalg import qr

np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(suppress=True, precision=5)


def Norm_Two(arr):
    # 求出二范数
    return math.sqrt(sum(arr**2))


def givens(A):

    pass


def cacl_x(x, y):
    return np.dot(x, y) / np.dot(y, y)


def fill_Q(arr):
    new_arr = [
        list(arr[i]) + [0] * (len(arr) - i - 1) for i in range(len(arr))[::-1]
    ]
    return np.array(new_arr)


def find_num(y, Q):
    num = 1
    for (temp, e) in zip(y, Q):
        if temp != 0 and e != 0:
            num = temp / e
            break
    return num


def schmidt(A):
    """
    Schmidt正交化
    """
    n, _ = A.shape
    col = 0
    y = []
    Q, R = [], []
    while col < n:

        # 保留关于y方程的y的系数
        coef_lst = []

        # 计算每个x
        x_col = A[:, col]
        # 计算每个y
        y_col = x_col.copy()
        # 将每个x进行正交化
        for y_item in y:
            temp_num = cacl_x(x_col.copy(), y_item)
            coef_lst.append(temp_num)
            temp_col = temp_num * np.array(y_item)
            y_col = y_col - temp_col
        y.append(y_col)

        # 计算每个e
        y_sum = 0
        for y_num in y_col:
            y_sum += y_num**2

        # 得出Q的每一列
        Q_col = (1 / math.sqrt(y_sum)) * y_col
        Q.append(Q_col)

        # 得出R的每一列
        R_col = []
        for (coef_item, y_item, e_col) in zip(coef_lst, y[:-1], Q):
            temp_item = coef_item * y_item
            R_col.append(find_num(temp_item, e_col))
        R_col.append(find_num(y_col, Q_col))

        R.append(np.array(R_col))

        col += 1
    # 补充上三角矩阵
    R = fill_Q(R)
    # 矩阵转置，得出QR
    R = np.array(R).T[:, ::-1]
    Q = np.array(Q).T
    return R, Q


def householder(A):
    # 获取n, m
    n, _ = A.shape
    Q = np.eye(n)
    H_lst = []
    A_col = A.copy()
    # 遍历每一列
    col = 0
    while col < n - 1:
        # 取出每一列
        x_col = A_col[:, col]
        # 取当前列的范数，规定如果第一个元素大于等于零，则为正，否则为负
        sigma_col = Norm_Two(x_col[:n - col])
        sigma_col = sigma_col if x_col[col] >= 0 else -sigma_col
        print("sigma_col:", sigma_col)
        # 计算当前列的u, 之前的元素换为0

        mu_col = x_col.copy()

        for i in range(col):
            mu_col[i] = 0.0
        mu_col[col] = sigma_col + mu_col[col]
        mu_col = np.array([mu_col]).T

        # 计算rho, rho=sigma * mu
        rho_col = sigma_col * mu_col[col]

        print("rho_col", rho_col)
        print("mu_col", mu_col)
        # 计算H, H=I- rho^{-1}*mu*mu.T
        H_col = np.eye(n) - np.dot(mu_col, mu_col.T) / rho_col
        H_lst.append(H_col)

        Q = np.dot(Q, H_col)
        # 计算A, 如果不进行int64操作，可能会导致特别小的数出现
        A_col = np.dot(H_col, A_col)
        A_col = np.int64(A_col)

        col += 1
        print("A_col:", A_col)
        print("H_col:", H_col)
    return Q, A_col


def householder_reflection(A):
    """Householder变换
    Householder变换, QR分解的一种方式
    # 公式
    $H = I - \frac{2}{v^Tv}vv^T$    
    $v = c + ||c||_2e$

    I是nxn的单位矩阵，v、c、e都是有n个元素的列向量
    ||c||_2是表示c的二范数


    $u = c - e$
    $v = u / ||u||_2$
    $H = I - 2 * (v.T * v)$

    $R = Q * H$
    $Q = H * R$
    具体步骤见代码:    
    """
    (r, c) = np.shape(A)
    Q = np.eye(r)
    R = np.copy(A)
    for cnt in range(r - 1):
        x = R[cnt:, cnt]
        e = np.zeros_like(x)
        e[0] = Norm_Two(x)
        u = x - e
        v = u / Norm_Two(u)
        Q_cnt = np.eye(r)
        Q_cnt[cnt:, cnt:] -= 2.0 * np.outer(v, v)
        print("v:", v)
        print("np.outer:", np.outer(v, v))
        R = np.dot(Q_cnt, R)  # R=H(n-1)*...*H(2)*H(1)*A
        Q = np.dot(Q, Q_cnt)  # Q=H(n-1)*...*H(2)*H(1)  H为自逆矩阵
    print(np.dot(Q, R))
    return (Q, R)


def test_schmidt():
    A = np.array([[0, 3, 1], [0, 4, -2], [2, 1, 2]])
    temp_A = A.copy()
    print(A)
    res = qr(temp_A)
    R, Q = schmidt(A)

    print("Q:\n", Q)
    print("R:\n", R)
    # print("A:\n", A)


def test_householder():
    # size = (4, 4)
    # arr为待分解的矩阵
    # A = np.array([[1, 1, 1], [2, -1, -1], [2, -4, 5]])
    A = np.array([[0, 3, 1], [0, 4, -2], [2, 1, 2]])
    A = np.array([[5, -3, 2], [6, -4, 4], [4, -4, 5]])
    temp_A = A.copy()
    print(A)
    householder_reflection(A)
    # res = qr(temp_A)

    # print("A:\n", A)

    Q, R = householder(temp_A)
    print("Q:\n", Q)
    print("R:\n", R)

    print(np.dot(Q, R))


# QR分解
if __name__ == '__main__':
    A = np.array([[1, 1, 1, 1], [2, -1, -1, 1], [2, -4, 5, 1]])
    # A = np.array([[0, 3, 1, 1], [0, 4, -2, 1], [2, 1, 2, 1]])
    # A = np.array([[5, -3, 2], [6, -4, 4], [4, -4, 5]])
    temp_A = A.copy()
    print(A)
    householder_reflection(A)
    # res = qr(temp_A)
    Q, R = householder(A)
    # R, Q = schmidt(A)
    # print(np.dot(Q, R))
    # print("Q:\n", Q)
    # print("R:\n", R)
    # print("A:\n", A)

    print(np.dot(Q, R))
