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

#declare list to store predicted and actual classes
confusion_list = []

confusion_matrix = [[]]

with open(args.JSONFILE.name) as f:
    d_tree_dict = json.load(f)

def predict_class_label(row, node):
    local_var = node['var'] #get edge with matching val in row[va] if leaf, return else, recurse
    for edge in node['edges']:
        if (edge['edge']["value"] == row[1][local_var]):
            if 'leaf' in edge['edge']:
                return edge['edge']['leaf']
            else:
                return predict_class_label(row, edge['edge']['node'])
    return None #if we got here tree is incomplete

for row in trainingSet.iterrows(): #predict value with decision tree and compare to true value
    p_class_struct = predict_class_label(row, d_tree_dict['node'])
    a_class = row[1][-1]
    confusion_list.append((p_class_struct, a_class))

def print_stats(confusion_list, predict_class_label):
    total_classified = len(confusion_list)
    total_classified_correct = 0
    total_classified_incorrect = 0
    confusion_lists = {} #each 
    for val in trainingSet.iloc[:,-1].unique(): #for each possible class label
        confusion_lists[val] = {}
    for a_val in confusion_lists: # for each possible actual value
        for val in trainingSet.iloc[:,-1].unique():
            confusion_lists[a_val][val] = 0

    for pair in confusion_list:
        confusion_lists[pair[1]][pair[0]['decision']] += 1
        #(predicted, actual)
        if (pair[0]['decision'] == pair[1]):
            total_classified_correct += 1
        else:
            total_classified_incorrect += 1
    print("total number of records classified: " + str(total_classified))
    print("total number of records correctly classified: " + str(total_classified_correct))
    print("total number of records incorrecly classified: " + str(total_classified_incorrect))
    print("overall accuracy: " + str(total_classified_correct/total_classified))
    print("error rate: " + str(total_classified_incorrect/total_classified))
    print("{actual_value[predicted, predicted, etc],etc}")
    print(confusion_lists)

print_stats(confusion_list, predict_class_label)
