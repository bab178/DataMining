# Data Mining Spring 2017 Assignment 2
# Blake Bordovsky
# python 3.6.0 using sklearn

import sys
import numpy
from sklearn.cluster import KMeans

def getDataFromFile(inputFile):
    with open(inputFile) as f:
        content = f.readlines()

    pairs = []
    for x in content:
        pair = x.strip("\n").split(" ")
        pairs.append([int(pair[0]), int(pair[1])])
    return pairs

k = int(sys.argv[1])
inputFile = sys.argv[2]
pairs = getDataFromFile(inputFile)

print("Processing", inputFile, "with", str(k), "clusters.")

X = numpy.array(pairs)
kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
labels = [l+1 for l in kmeans.labels_]

orig_stdout = sys.stdout
o = open('output.txt', 'w')
for i in range(len(pairs)):
    o.write("{} {} {}\n".format(pairs[i][0], pairs[i][1], labels[i]))
o.close()

print("Completed: Wrote output to output.txt")