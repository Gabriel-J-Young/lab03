import math
from anytree import Node, RenderTree
import json
import pandas as pd


def predict_class_label(row, node):
    # return the predicton for the row
    #if the node has an edge with a matching attribue that leads to a leaf, return the value of that leaf, else recurse
    row_decision_attr = row[1][node['var']] 
    for edge_container in node['edges']:
        edge = edge_container['edge']
        edge_value = edge['value']
        condition = '=='
        if edge['value'][0] == '>':
            condition = '>='
            real_value = edge['value'][2:len(edge['value'])]
            edge_value = real_value
        if edge['value'][0] == '<':
            condition = '<'
            real_value = edge['value'][1:len(edge['value'])]
            edge_value = real_value

        if (condition == '==' and   str(row_decision_attr) == str(edge_value)) or \
           (condition == '>=' and float(row_decision_attr) >= float(edge_value)) or \
           (condition ==  '<' and float(row_decision_attr)  < float(edge_value)):
            if 'leaf' in edge:
                return edge['leaf']
            else:
                return predict_class_label(row, edge['node'])
    return None #if we got here tree is incomplete

def print_stats(confusion_list, trainingSet):
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

def get_stats(confusion_list, trainingSet):
    stats = {}
    stats['total_classified'] = len(confusion_list)
    stats['total_classified_correct'] = 0
    stats['total_classified_incorrect'] = 0
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
            stats['total_classified_correct'] += 1
        else:
            stats['total_classified_incorrect'] += 1
    stats['accuracy'] = stats['total_classified_correct']/stats['total_classified']
    stats['error_rate'] = stats['total_classified_incorrect']/stats['total_classified']
    stats['confusion_mat'] = confusion_lists
    return stats