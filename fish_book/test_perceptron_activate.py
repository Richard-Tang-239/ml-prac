import matplotlib.pylab as plt
import numpy as np
# Test for perceptron
from common.perceptron_functions import AND, NAND, OR, XOR
from fish_source.dataset.mnist import load_mnist

print("AND:")
print(str(AND(0, 0)))
print(str(AND(0, 1)))
print(str(AND(1, 0)))
print(str(AND(1, 1)))

print("\nOR:")
print(str(OR(0, 0)))
print(str(OR(0, 1)))
print(str(OR(1, 0)))
print(str(OR(1, 1)))

print("\nNAND:")
print(str(NAND(0, 0)))
print(str(NAND(0, 1)))
print(str(NAND(1, 0)))
print(str(NAND(1, 1)))

print("\nXOR:")
print(str(XOR(0, 0)))
print(str(XOR(0, 1)))
print(str(XOR(1, 0)))
print(str(XOR(1, 1)))


# Test for activate_functions
from common.activate_functions import relu, sigmoid, softmax, step

x = np.arange(-5, 5, 0.5)  
y_step = step(x)
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_softmax = softmax(x)  # np.array([0.3, 2.9, 4.0])
plt.plot(x, y_step)
plt.plot(x, y_sigmoid)
plt.plot(x, y_relu)
plt.plot(x, y_softmax)
plt.ylim(-0.1, 1.1)
plt.show()
