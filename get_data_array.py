from numpy import array
from random import shuffle


def get_data_array(filename):
    with open(filename, "r") as dataFile:
        dataList = dataFile.read().splitlines()

    data = [d.split("\t") for d in dataList]
    # need to shuffle!
    shuffle(data)
    shuffle(data)
    shuffle(data)

    X = []
    for word in data:
        X0 = []
        for vec in word[:-1]:
            X0 += vec.split(" ")
        X0 = array(X0)
        X.append(X0)
    Y = [d[-1] for d in data]

    Y = array(Y)
    X = array(X)

    return X, Y
