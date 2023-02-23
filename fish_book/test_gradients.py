import numpy as np
import matplotlib.pyplot as plt

from common.gradient_functions import gradient_decent


def function_1(x):
    return np.sum(x**2)


init_x = np.array([8., 9.5])
lr = 0.1
step_num = 100
local_minimal, x_history = gradient_decent(function_1, init_x, lr, step_num)
print("The local minimal of gradient decent: " + str(local_minimal))

# 画出每次迭代的变化值
plt.plot([-10, 10], [0, 0], '--b')
plt.plot([0, 0], [-10, 10], '--b')
plt.plot(x_history[:, 0], x_history[:, 1], 'o')
plt.xlim(-10, 10)
plt.xlim(-10, 10)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
