from scipy.io import loadmat
from scipy.io import mmread,mminfo
from scipy.sparse import csr_matrix
import sys
import matplotlib.pyplot as plt
import random, math
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def readEmbeddings(filename, nodes, dim):
    embfile = open(filename, "r")
    firstline = embfile.readline()
    N = int(firstline.strip().split()[0])
    X = [[0]*dim for i in range(nodes)]
    for line in embfile.readlines():
        tokens = line.strip().split()
        nodeid = int(tokens[0])-1
        x = []
        for j in range(1, len(tokens)):
            t = float(tokens[j])
            x.append(t)
        X[nodeid] = x
    embfile.close()
    print("Size of X:", len(X))
    return np.array(X)

def readEmbeddingsHARP(filename, nodes):
    dX = np.load(filename)
    print("Size of X:", len(dX))
    return dX

filename = sys.argv[2]
nodes = int(sys.argv[3])
algoname = sys.argv[4]

if sys.argv[1] == "1":
    X = readEmbeddings(filename, nodes, 128) 
else:
    X = readEmbeddingsHARP(filename, nodes)

output = open(algoname+".txt", "w")
i = 1
for coord in X:
    for t in coord:
        output.write(str(t) + "\t")
    output.write("\n")
    i += 1
output.close()
