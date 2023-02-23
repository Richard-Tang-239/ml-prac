import numpy as np
import matplotlib.pylab as plt

from common.TwoLayerNet import TwoLayerNet
from fish_source.dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

train_loss_list = []

# 超参数
iter_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

# 初始化神经网络
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iter_num):
    # 获取 min_batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    # grads = network.numerical_gradient(x_batch, t_batch)
    grads = network.gradient(x_batch, t_batch)

    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grads[key]

    # 记录学习过程
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

x_series = np.arange(1, iter_num + 1, 1)
plt.plot(x_series, train_loss_list)
plt.show()
