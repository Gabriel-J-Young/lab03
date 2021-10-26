import argparse
import math
import pandas as pd
import json
import os
from anytree import Node, RenderTree
from six import assertRaisesRegex
from c45 import C45
from c45 import tree_to_json

gain_threshold = 0.2

parser = argparse.ArgumentParser()
parser.add_argument('TrainingSetFile',
                    type = argparse.FileType('r'))
parser.add_argument('restrictionsFile',
                    nargs='?',
                    type = argparse.FileType('r'))
args = parser.parse_args()

trainingSet = pd.read_csv(args.TrainingSetFile.name, skiprows=[1,2])

restrict_list = None
if args.restrictionsFile != None:
    my_file = open(args.restrictionsFile.name, "r")
    content = my_file.read()
    restrict_list = content.split(",")
    my_file.close()

def restricted_dataset(D,A, restrict_list):
    for i in range(0,len(restrict_list)):
        if int(restrict_list[i])==1:
            D = D.drop(columns = [A[i]])
            del A[i]
    return D

A = []
for attribute in trainingSet.columns.values:
    A.append(attribute)
del A[-1]

if restrict_list != None:
    trainingSet = restricted_dataset(trainingSet, A, restrict_list)

#print(trainingSet)

T = C45(trainingSet, gain_threshold)
#print(RenderTree(T))
name = args.TrainingSetFile.name
j = tree_to_json(T, name)
json_data_file = open(os.path.splitext(args.TrainingSetFile.name)[0] + ".json", "w")
json_data_file.write(j)
json_data_file.close()

