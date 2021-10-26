#File takes JSON decision tree and training data to be classified as input

import argparse
import json
import pandas as pd
from anytree import Node, RenderTree
from six import print_
from classifyAlg import predict_class_label
from classifyAlg import print_stats

parser = argparse.ArgumentParser()
parser.add_argument('CSVFile',
                    type = argparse.FileType('r'))

parser.add_argument('JSONFILE',
                    type = argparse.FileType('r'))
args = parser.parse_args()

#load csv training data and json decision tree

trainingSet = pd.read_csv(args.CSVFile.name, skiprows=[1,2])

#declare list to store predicted and actual classes
confusion_list = []



with open(args.JSONFILE.name) as f:
    d_tree_dict = json.load(f)



for row in trainingSet.iterrows(): #predict value with decision tree and compare to true value
    p_class_struct = predict_class_label(row, d_tree_dict['node'])
    a_class = row[1][-1]
    confusion_list.append((p_class_struct, a_class))

print_stats(confusion_list, trainingSet)
