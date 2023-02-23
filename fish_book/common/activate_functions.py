import numpy as np


def identity(x):
    return x


def step(x):
    y = x > 0
    return y.astype(np.int32)
    

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


'''
def softmax(x):
    # 减去输入信号的最大值，防止指数函数计算溢出
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x
'''


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))
