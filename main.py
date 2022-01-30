import threading
import pandas as pd
import Tree as tree
import re
import timeit
import numpy as np
import RandomForest as rf
from collections import Counter
import json

def unique(list1):
    x = np.array(list1)
    return np.unique(x)

def main():
    f = open("config.json", "r")
    config = json.load(f)
    f.close()
    unknown = config["unknown"]
    train_data_path = config["train_data_path"]
    test_data_path = config["test_data_path"]
    forest_size = config["forest_size"]
    max_depth = config["max_depth"]
    delimeter = config["delimeter"]
    class_column = config["class_column"]

    train_data = pd.read_csv(train_data_path, header=None, delimiter=delimeter)
    test_data = pd.read_csv(test_data_path, header=None, delimiter=delimeter)
    size = len(train_data.axes[1])
    for r in train_data:
        for l in train_data[r]:
            if (re.sub(r"\s+", "", str(l)) == unknown):
                train_data.drop(train_data[train_data[r] == l].index, inplace=True)

    for r in test_data:
        for l in test_data[r]:
            if (re.sub(r"\s+", "", str(l)) == unknown):
                test_data.drop(test_data[test_data[r] == l].index, inplace=True)
    forest_size = forest_size
    Y = train_data[class_column].values.tolist()

    forest = []
    threads = list()
    start = timeit.default_timer()
    train_data = train_data.drop(train_data.columns[class_column], axis=1)
    random_forest = rf.RandomForest(forest_size)
    for index in range(0, forest_size):
        root = tree.Node(Y, train_data, max_depth=max_depth)
        random_forest.trees.append(root)
        x = threading.Thread(target=root.grow_tree, args=())
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        thread.join()

    stop = timeit.default_timer()

    print('Time: ', stop - start)

    random_forest.predict_set(test_data, unique(Y))
    #tree.predict_set(test_data, forest, unique(Y))


main()
