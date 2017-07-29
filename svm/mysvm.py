
def hinge(x):
    k = 1 - x
    return k if k > 0 else 0


def train(dataset, labels, epoch=1000, learning_rate=0.01):
    ds_size = len(dataset)
    feature_size = len(dataset[0])
    weights = [1.0] * feature_size
    for epoch_index in range(epoch):
        for i in range(ds_size):
            instance = dataset[i]
            label = labels[i]
            grads = [0.0] * feature_size
            wx = sum([w * x for (w, x) in zip(weights, instance)])
            if 1 - wx * label > 0:
                for j in range(feature_size):
                    grads[j] = weights[j] - instance[j] * label
            else:
                grads[feature_size-1] = weights[feature_size-1]

            for j in range(feature_size):
                weights[j] -= learning_rate * grads[j]
            print("current weights: {}".format(weights))
    return weights


def predict(weights, instance):
    wx = sum([w * x for (w, x) in zip(weights, instance)])
    predict_value = 1 if wx > 0 else -1
    print("predict instance: {} label: {}".format(instance, predict_value))
    return predict_value

if __name__ == '__main__':
    # [4, 5]
    dataset = [[2.0, 3.0, 0.0, 5.0, 2.8], [3.0, 4.0, 5.0, 0.0, 0.0],
               [0.5, 0.3, 0.7, 0.9, 0.0], [-2.0, 0.1, 0.2, 0.9, 0.0]]
    # [4]
    labels = [1, 1, -1, -1]
    model = train(dataset, labels)

    test_instance = [2.0, 3.0, 0.0, 5.0, 2.8]
    predict(model, test_instance)
    test_instance = [0.5, 0.3, 0.7, 0.9, 0.0]
    predict(model, test_instance)
