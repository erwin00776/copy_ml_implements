import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def logistic_regssion(epoch=500, learning_rate=0.01):
    train_features = np.array([1, 0, 25], dtype=np.float)
    train_labels = np.array([0], dtype=np.int)
    weights = np.ones(len(train_features), dtype=np.float)

    for i in range(epoch):
        multiple_result = train_features * weights
        train_predicts = sigmoid(np.sum(multiple_result))
        train_diff = train_labels - train_predicts

        grad = train_features * train_diff
        weights += learning_rate * grad
        print("weights is: {}".format(weights))

        preds = sigmoid(np.sum(train_features * weights))
        print("predict true probability is: {}".format(preds))


if __name__ == '__main__':
    logistic_regssion()
