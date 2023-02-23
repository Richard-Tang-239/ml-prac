import numpy as np


def cross_entropy_error(y, t):
    """
    交叉熵损失函数

    Parameters
    ----------
    y : 输入数据
    t : 监督数据

    Returns
    -------
    error : 交叉熵误差
    if y.ndim == 1:
        t = t.reshape(1, t.size)
    error = -np.sum(t * np.log(y + 1e-7)) / t.shape[0]
    #print("Cross entropy error: " + str(error))
    return error
    
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def mean_squared_error(y, t):
    """
    均方差损失函数

    Parameters
    ----------
    y : 输入数据
    t : 监督数据

    Returns
    -------
    error : 均方差误差
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
    error = 0.5 * np.sum((y - t)**2) / t.shape[0]
    #print("Mean squared error: " + str(error))
    return error
