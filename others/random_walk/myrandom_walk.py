import random
import matplotlib.pyplot as plt


def random_walk():
    value = random.randint(0, 100)
    values = [value]
    epoch = 10000
    for i in range(epoch):
        if random.randint(0, 1) == 1:
            value += 1
        else:
            value -= 1
        values.append(value)
    plt.plot(range(len(values)), values)
    plt.show()

if __name__ == '__main__':
    random_walk()