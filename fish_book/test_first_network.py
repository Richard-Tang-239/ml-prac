import numpy as np
from common.activate_functions import sigmoid
from common.activate_functions import identity

def init_first_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

def first_forward(network, x):
    # 输入层 -> 第一层
    a1 = sigmoid(np.dot(x, network['W1']) + network['b1'])
    # 第一层 -> 第二层
    a2 = sigmoid(np.dot(a1, network['W2']) + network['b2'])
    # 第二层 -> 输出层
    y = identity(np.dot(a2, network['W3']) + network['b3'])
    return y


network = init_first_network()
x = [1.0, 0.5]
y = first_forward(network, x)
print(y)  # [0.31682708 0.69627909]

# 一般输出层设计：
# 回归问题使用恒等函数（identity）
# 二元分类问题使用 sigmoid 函数
# 多元分类问题使用 softmax 函数
