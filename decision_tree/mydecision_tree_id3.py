import math
import operator
import json


def calc_shannon_entropy(dataset):
    dataset_size = len(dataset)
    class_count = {}
    for instance in dataset:
        label = instance[-1]
        count = class_count.get(label, 0)
        class_count[label] = count + 1
    shannon_entropy = 0.0
    for clazz in class_count.values():
        probability = float(clazz) / dataset_size
        shannon_entropy += -1.0 * probability * math.log(probability, 2)
    return shannon_entropy


def mytest_calc_shannon_entropy():
  # Should be 1.0
  dataset = [[0, 0, 0, 0, 'N'], [0, 0, 1, 1, 'Y']]
  print("The shannon entropy is: {}".format(calc_shannon_entropy(dataset)))

  # Should be 0.0
  dataset = [[0, 0, 0, 0, 'N'], [0, 0, 1, 1, 'N']]
  print("The shannon entropy is: {}".format(calc_shannon_entropy(dataset)))


def split_dataset(dataset, feature, value):
    new_dataset = []
    for instance in dataset:
        if instance[feature] == value:
            new_instance = instance[: feature]
            new_instance.extend(instance[feature + 1:])
            new_dataset.append(new_instance)
    return new_dataset


def choose_best_feature_to_split(dataset):
    feature_counts = len(dataset[0]) - 1
    best_feature = -1
    best_after_split_entropy = 0.0
    for i in range(feature_counts):
        instance_with_one_feature = [instance[i] for instance in dataset]
        feature_value_set = set(instance_with_one_feature)
        after_split_entropy = 0.0
        for value in feature_value_set:
            subset = split_dataset(dataset, i, value)
            probability = len(subset) / float(len(dataset))
            after_split_entropy += probability * calc_shannon_entropy(subset)
        if after_split_entropy > best_after_split_entropy:
            best_after_split_entropy = after_split_entropy
            best_feature = i
    return best_feature


def create_decision_tree(dataset, header_names):
    labels = [instance[-1] for instance in dataset]
    if labels.count(labels[0]) == len(labels):
        return labels[0]
    if len(dataset[0]) == 1:
        label_count_map = {}
        for label in labels:
            if label not in label_count_map.keys():
                label_count_map[label] = 0
            label_count_map[label] += 1
        sorted_label_count_map = sorted(
            label_count_map.iteritems(), key=operator.itemgetter(1), reversed=True)
        return sorted_label_count_map[0][0]

    best_feature_id = choose_best_feature_to_split(dataset)
    root_name = header_names[best_feature_id]
    del header_names[best_feature_id]

    decision_tree = {root_name: {}}
    feature_values = [instance[best_feature_id] for instance in dataset]
    unique_feature_values = set(feature_values)
    for value in unique_feature_values:
        sub_header_names = header_names[:]
        sub_dataset = split_dataset(dataset, best_feature_id, value)
        subtree = create_decision_tree(sub_dataset, sub_header_names)
        decision_tree[root_name][value] = subtree
    return decision_tree


def predict(decision_tree, test_header_names, test_dataset):
    root_key = decision_tree.keys()[0]
    subtree = decision_tree[root_key]

    feature_index = test_header_names.index(root_key)
    for key in subtree.keys():
        if test_dataset[feature_index] == key:
            if type(subtree[key]).__name__ == 'dict':
                predict_label = predict(subtree[key], test_header_names, test_dataset)
            else:
                predict_label = subtree[key]
    return predict_label


if __name__ == '__main__':
    dataset = [[0, 0, 0, 0, 'N'], [0, 0, 0, 1, 'N'], [1, 0, 0, 0, 'Y'],
               [2, 1, 0, 0, 'Y'], [2, 2, 1, 0, 'Y'], [2, 2, 1, 1, 'N'], [1, 2, 1, 1, 'Y']]
    header_names = ['outlook', 'temperature', 'humidity', 'windy']

    # mytest_calc_shannon_entropy()
    decision_tree = create_decision_tree(dataset, header_names)
    # print(decision_tree)
    print(json.dumps(decision_tree, indent=2))

    # Test
    test_header_names = ['outlook', 'temperature', 'humidity', 'windy']
    test_dataset = [2, 1, 0, 0]
    result = predict(decision_tree, test_header_names, test_dataset)
    print("Predict decision tree and get result: {}".format(result))

    test_dataset1 = [2, 2, 1, 1]
    print("Predict decision tree and get result: {}".format(predict(decision_tree,
                                                                    test_header_names,
                                                                    test_dataset1)))
