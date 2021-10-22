import argparse
import math
import pandas as pd


def knn(D, k, x):
    # D - dataset
    # k - number of nearest neighbors
    # x - point to classify
    for d in D:
        dist[d] = distance(d, x) #compute distance


