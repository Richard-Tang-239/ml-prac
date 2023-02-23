from PIL import Image
import numpy as np
import pickle
from fish_source.dataset.mnist import load_mnist
from common.activate_functions import sigmoid
from common.activate_functions import softmax
from common.loss_functions import cross_entropy_error
from common.loss_functions import mean_squared_error


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def init_network():
    with open("fish_book/fish_source/ch03/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = sigmoid(np.dot(x, W1) + b1)
    a2 = sigmoid(np.dot(a1, W2) + b2)
    y = softmax(np.dot(a2, W3) + b3)
    return y


def predict_all_test_data(network, x_test, l_test):
    batch_size = 100
    accuracy_cnt = 0
    for i in range(0, len(x_test), batch_size):
        x_batch = x_test[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p_batch = np.argmax(y_batch, axis=1)
        predict_l_batch = np.argmax(l_test[i:i+batch_size], axis=1)
        accuracy_cnt += np.sum(p_batch == predict_l_batch)
    print("Accuracy: " + str(float(accuracy_cnt) / len(x_test)))


# 加载训练数据
# flatten: 将数据从0~255的区间转换成0~1，方式指数函数计算时溢出
# one_hot_label:
(x_train, l_train), (x_test, l_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)
network = init_network()

# 查看测试样本的第一张图片
img = x_test[0].reshape(28, 28) * 255
img_show(img)

predict_all_test_data(network, x_test, l_test)

# min-batch training
batch_size = 100
train_size = x_train.shape[0]
batch_mask = np.random.choice(train_size, batch_size)
l_batch = l_train[batch_mask]
x_batch = x_train[batch_mask]
y_batch = predict(network, x_batch)
mean_squared_error(y_batch, l_batch)
cross_entropy_error(y_batch, l_batch)
