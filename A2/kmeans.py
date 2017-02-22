# Data Mining Spring 2017
# Blake Bordovsky

import os
import sys
from sklearn.cluster import KMeans
import numpy as np

def getFileNames(root):
    ls = []
    for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f.endswith(".txt")]:
            ls.append(os.path.join(dirpath, filename))
    return ls

def getDataFromFile(inputFile):
    with open(inputFile) as f:
        content = f.readlines()

    k = int(content.pop(0))
    pairs = []
    for x in content:
        pair = x.strip("\n").split(" ")
        pairs.append([int(pair[0]), int(pair[1])])
    return [k, pairs]


files = getFileNames(".")


for inputFile in files:
    fileNum = inputFile[-5]
    data = getDataFromFile(inputFile)
    k = data[0]
    pairs = data[1]

    X = np.array(pairs)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    labels = [l+1 for l in kmeans.labels_]

    outFileName = 'output' + fileNum + '.txt'

    orig_stdout = sys.stdout
    o = open(outFileName, 'w')
    sys.stdout = o

    for i in range(len(pairs)):
        print(pairs[i][0], pairs[i][1], labels[i])

    sys.stdout = orig_stdout
    o.close()

    print(outFileName, " completed")

    



