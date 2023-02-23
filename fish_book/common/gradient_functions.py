import numpy as np

'''
def numerical_gradient(f, x):
    """
    数值微分

    Parameters
    ----------
    f : 原始函数
    x : 微分参数

    Returns
    -------
    grads : 数值微分结果
    """
    h = 1e-4 #0.0001
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_value = x[idx]
        # f(x+h)
        x[idx] = tmp_value + h
        fxh1 = f(x)
        # f(x-h)
        x[idx] = tmp_value - h
        fxh2 = f(x)
        # 计算偏导数
        grad[idx] = (fxh1 - fxh2) / (2*h)
        # 还原值
        x[idx] = tmp_value

    return grad
'''

def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val  # 还原值
        it.iternext()

    return grad

def gradient_decent(f, init_x, lr=0.01, step_num=100):
    """
    梯度下降

    Parameters
    ----------
    f : 要进行最优化的函数
    init_x : 初始值
    lr : 学习率
    step_number : 迭代次数

    Returns
    -------
    x : 最优解结果
    x_history : 迭代历史
    """
    x = init_x
    x_history = []
    for i in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x, np.array(x_history)