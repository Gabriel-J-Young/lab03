#File takes JSON decision tree and training data to be classified as input

import argparse
import math
import json
import pandas as pd
from anytree import Node, RenderTree

parser = argparse.ArgumentParser()
parser.add_argument('CSVFile',
                    type = argparse.FileType('r'))

parser.add_argument('JSONFILE',
                    type = argparse.FileType('r'))
args = parser.parse_args()

#load csv training data and json decision tree

trainingSet = pd.read_csv(args.CSVFile.name, skiprows=[1,2])

with open(args.JSONFILE.name) as f:
    d_tree_dict = json.load(f)
#print(d_tree_dict)
#print(d_tree_dict['node']['var'])
#print(d_tree_dict['node']['edges'][''])

def predict_class_label(row, node):
    local_var = node['var'] #get edge with matching val in row[va] if leaf, return else, recurse
    for edge in node['edges']:
        if (edge['edge']["value"] == row[1][local_var]):
            if 'leaf' in edge['edge']:
                return edge['edge']['leaf']
            else:
                return predict_class_label(row, edge['edge']['node'])
    return None #if we got here tree is incomplete

for row in trainingSet.iterrows(): #predict value with decision tree and compair to true value
    print(row)
    print(predict_class_label(row, d_tree_dict['node']))