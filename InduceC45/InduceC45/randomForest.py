import argparse
import pandas as pd
import numpy as np
import random
import sys
import json
from collections import Counter
from anytree import Node, RenderTree
from c45 import C45
from c45 import tree_to_json
from classifyAlg import predict_class_label
from classifyAlg import print_stats
from classifyAlg import get_stats
import time

gain_threshold = 0.0 # no 
num_folds = 3

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

def get_forest(D,args):
    A = []
    for attribute in D.columns.values:
        A.append(attribute)
    classifier = A[-1]
    del A[-1]
    forest = []
    for i in range(args.num_trees):
        rand_dataset = get_rand_dataset(D,A,args.num_attributes, args.num_data_points)
        tree = C45(rand_dataset, gain_threshold)
        name = args.training_set_file.name
        forest.append(tree_to_json(tree, name))
    return forest


def random_predicted(D):
    #return a random element from the classifier of D
    class_row = D.iloc[:,-1:]
    choice = class_row.sample()
    return str(choice.iloc[0][0])

def get_confusion(D, forest):
    #given a dataset and a forest, returns a confusion matrix
    confusion_list = []
    for row in D.iterrows(): #predict value with decision tree and compare to true value
        decisions = []
        for tree in forest:
            predicted_struct = predict_class_label(row, tree['node'])
            if predicted_struct is not None: 
                decisions.append(predicted_struct['decision'])
            else:
                decisions.append(random_predicted(D))
        a_class = row[1][-1]
        if str(type(a_class)) == "<class 'numpy.float64'>":
            a_class = int(a_class)
        c = Counter(decisions)
        decision = c.most_common(1)[0][0]
        p_class_struct = {'decision':decision} # to match classify methods
        confusion_list.append((p_class_struct, str(a_class)))
    return confusion_list

def sum_mat(overall_confusion_matrix, c_mat):
    if (overall_confusion_matrix == None): #if overall_confusion_matrix is empty set it to c_mat
        overall_confusion_matrix = c_mat.copy()
    else: #else loop through c_mat and accumulate values
        
        for actual in c_mat:
            for key, value in c_mat[actual].items():
                overall_confusion_matrix[actual][key] += value
    return overall_confusion_matrix

def print_c_mat(mat):
    print(",", end="")
    for key, val in mat.items():
        print(key, end=",")
    print()
    for key, val in mat.items():
        print(key, end=",")
        for key_in, val_in in val.items():
            print(val_in, end=",")
        print()

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

big_training_set = pd.read_csv(args.training_set_file.name, skiprows=[1,2])
A = []
for attribute in big_training_set.columns.values:
    A.append(attribute)
classifier = A[-1]
del A[-1]

if args.num_attributes > len(A):
    sys.stderr.write("num_attributes too large\n")
    raise ValueError
    
# now time to classify- forest has all the random trees




total_correct = 0
total_classified = 0


#slice dataset into n slices of equal size
shuffled = big_training_set.sample(frac=1)
slices = np.array_split(shuffled, num_folds)

overall_confusion_mat = None
total_classified = len(big_training_set.index)
total_correct = 0
total_incorrect = 0
random_predicted(big_training_set)
for idx, slice in enumerate(slices): #each slice must be designated as a holdout set once
    training_slices = slices[0:idx] + slices[idx+1:]
    training_set = pd.concat(training_slices)
    start_time = time.time()
    forest = get_forest(training_set, args)
    end_time = time.time()
    print("time:", idx, end_time-start_time)

    #now time to classify
    testing_set = slices[idx]
    
    confusion_list = get_confusion(testing_set, forest)
    stats = get_stats(confusion_list, testing_set)

    overall_confusion_mat = sum_mat(overall_confusion_mat, stats['confusion_mat'])
    total_correct += stats['total_classified_correct']
    total_incorrect += stats['total_classified_incorrect']
print("Overall accuracy: " + str(total_correct/total_classified))
print("Overall error: " + str(total_incorrect/total_classified))

print_c_mat(overall_confusion_mat)
