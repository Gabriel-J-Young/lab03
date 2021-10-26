import argparse
import pandas as pd
import random
import sys
import json
from collections import Counter
from anytree import Node, RenderTree
from c45 import C45
from c45 import tree_to_json
from classifyAlg import predict_class_label
from classifyAlg import print_stats

gain_threshold = 0.0 # no pruning

def restrict_dataset(D,A, restrict_list):
    restricted_D = D.copy()
    for removed_attr in restrict_list:
        testricted_D = restricted_D.drop(columns = [A[i]])
    for i in range(0,len(restrict_list)):
        if int(restrict_list[i])==1:
            D = D.drop(columns = [A[i]])
            del A[i]
    return D
    


def get_rand_dataset(D, A, num_attrs, num_dp):
    #return a new dataframe with randomly selected attributes and datapoints
    rand_A = random.sample(A, num_attrs) #selecting WITHOUT replacement
    attrs_to_drop = A.copy() # inverse of rand_A
    for attr in rand_A:
        attrs_to_drop.remove(attr)
    rand_attr_dataset = D.copy().drop(columns = attrs_to_drop)
    rand_dataset = rand_attr_dataset.sample(n=num_dp, replace=True)
    return rand_dataset


parser = argparse.ArgumentParser()
parser.add_argument('training_set_file',
                    type = argparse.FileType('r'))
parser.add_argument('num_attributes',
                    type = int)
parser.add_argument('num_data_points',
                    type = int)
parser.add_argument('num_trees',
                    type = int)
args = parser.parse_args()

#TODO: add arg checking 

training_set = pd.read_csv(args.training_set_file.name, skiprows=[1,2])
A = []
for attribute in training_set.columns.values:
    A.append(attribute)
classifier = A[-1]
del A[-1]

if args.num_attributes > len(A):
    sys.stderr.write("num_attributes too large\n")
    raise ValueError
forest = []
for i in range(args.num_trees):
    rand_dataset = get_rand_dataset(training_set,A,args.num_attributes, args.num_data_points)
    tree = C45(rand_dataset, gain_threshold)
    name = args.training_set_file.name
    forest.append(tree_to_json(tree, name))
    
# now time to classify

#declare list to store predicted and actual classes
confusion_list = []
for row in training_set.iterrows(): #predict value with decision tree and compare to true value
    decisions = []
    for tree in forest:
        predicted_struct = predict_class_label(row, tree['node'])
        #print(" ",predicted_struct['decision'])
        decisions.append(predicted_struct['decision'])
    c = Counter(decisions)
    decision = c.most_common(1)[0][0]
    #print("decision:", decision)
    a_class = row[1][-1]
    p_class_struct = {'decision':decision} # to match classify methods
    confusion_list.append((p_class_struct, a_class))

print_stats(confusion_list, training_set)    
