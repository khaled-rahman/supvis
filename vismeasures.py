from scipy.io import loadmat
from sklearn.utils import shuffle as skshuffle
from scipy.io import mmread,mminfo
from scipy.sparse import csr_matrix
import sys
import matplotlib.pyplot as plt
import networkx as nx
import random, math
import numpy as np
from sklearn.manifold import TSNE
from matplotlib.pyplot import cm
from matplotlib import collections as mc
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

def readEmbeddingsHARP(filename, nodes, dim):
    dX = np.load(filename)
    print("Size of X:", len(dX))
    return dX

def readgroundtruth(truthlabelsfile, N):
    Yd = dict()
    distinctlabels = set()
    lfile = open(truthlabelsfile)
    arrY = [-1 for i in range(N)]
    for line in lfile.readlines():
        tokens = line.strip().split()
        node = int(tokens[0])-1
        label = int(tokens[1])
        arrY[node] = label
        if label in Yd:
            tempy = Yd[label]
            tempy.append(node)
            Yd[label] = tempy
        else:
            Yd[label] = [node]
        distinctlabels.add(label)
    lfile.close()
    return Yd, len(distinctlabels), np.array(arrY)

def drawGraphc(G, X, comm, nl, algo1="Graph"):
    gridsize = (1, 1)
    fig = plt.figure(figsize=(8, 5))
    axIN = plt.subplot2grid(gridsize, (0, 0))
    plt.axis('off')
    axIN.set_xlim(min(X[:,0]), max(X[:,0]))
    axIN.set_ylim(min(X[:,1]), max(X[:,1]))
    linesIN = []
    e = 0
    print("Cluster:",len(comm))
    mycolors = cm.rainbow(np.linspace(0,1,nl+2))
    gd = dict()
    for com in comm:
        for node in list(comm[com]):
            gd[node] = com
            plt.scatter(X[node][0], X[node][1], s=2, color = mycolors[com])
    plt.axis('off')
    plt.savefig(algo1+'_vis.pdf')

filename = sys.argv[1]
G = mmread(filename)
graph = nx.Graph(G)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import metrics

if sys.argv[2] == "1":
    print("Running native...")
    X = readEmbeddings(sys.argv[3],  mminfo(filename)[0], int(sys.argv[4]))
else:
    X = readEmbeddingsHARP(sys.argv[3], mminfo(filename)[0], int(sys.argv[4]))

labs, l, gy = readgroundtruth(sys.argv[5], mminfo(filename)[0])
algoname = sys.argv[6]
print("Running TSNE")


if int(sys.argv[4]) == 2:
    print("Direct 2D Visualization")
    X_f = X
else:
    print("Visualization by TSNE")
    X_embedded = TSNE(n_components=2).fit_transform(X)
    X_f = X_embedde

drawGraphc(G, X_f, labs, l, algoname)

shil = metrics.silhouette_score(X_f, gy)
davd = metrics.davies_bouldin_score(X_f, gy)

print("silhouette:", shil, "davies_bouldin:", davd)

print("Visualization complete!")
