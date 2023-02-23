import numpy as np


def perceptron(x1, x2, w1, w2, b):
    x = np.array([x1, x2])
    w = np.array([w1, w2])
    value = np.dot(x, w) + b
    if (value > 0):
        return 1
    else:
        return 0


def AND(x1, x2):
    w1, w2, b = 0.5, 0.5, -0.7
    return perceptron(x1, x2, w1, w2, b)


def OR(x1, x2):
    w1, w2, b = 0.5, 0.5, 0
    return perceptron(x1, x2, w1, w2, b)


def NAND(x1, x2):
    w1, w2, b = -0.5, -0.5, 0.7
    return perceptron(x1, x2, w1, w2, b)


def XOR(x1, x2):
    '''使用2层感知机来实现异或门 XOR = AND(OR, NAND)
    '''
    s1 = OR(x1, x2)
    s2 = NAND(x1, x2)
    return AND(s1, s2)