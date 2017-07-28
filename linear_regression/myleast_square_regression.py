import numpy as np


def least_square_regression(epoch=1000, learning_rate=0.01):
    # y = 2 * x + 3
    x_array = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_array = [5.0, 7.0, 9.0, 11.0, 13.0]

    w = 1.0
    b = 1.0
    train_n = len(x_array)
    for i in range(epoch):
        w_grad = 0.0
        b_grad = 0.0
        loss = 0.0
        for j in range(train_n):
            x = x_array[j]
            y = y_array[j]
            w_grad -= 2.0 / train_n * (y - w * x - b) * x
            b_grad -= 2.0 / train_n * (y - w * x - b)
            loss += 1.0 / train_n * pow(y - w * x - b, 2)
        w -= learning_rate * w_grad
        b -= learning_rate * b_grad
        print("epoch: {}, w: {}, b: {}, loss: {}".format(i, w, b, loss))

if __name__ == '__main__':
    least_square_regression()