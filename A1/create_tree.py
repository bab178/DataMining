from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import os
import os.path
import subprocess
import numpy as np
from pathlib import WindowsPath

def visualize_tree(tree):
    with open("tree.dot", 'w') as f:
        export_graphviz(tree, out_file=f)

    command = ["C:/Anaconda3/Library/bin/graphviz/dot.exe", "-Tpng", "tree.dot", "-o", "tree.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not produce visualization of tree, reconfigure path to graphviz dot")


def getDataFilesFromFolder (root):
    files = []
    for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f.endswith(".csv")]:
            files.append(os.path.join(dirpath, filename))
    return files

files = getDataFilesFromFolder("./Datasets/")

for csv_file in files:
    dataset = np.genfromtxt(csv_file, delimiter=',')[:,:-1]
    print(dataset.shape)

# clf = DecisionTreeClassifier()
# clf = clf.fit(dataset.data, dataset.target)
# visualize_tree(clf)