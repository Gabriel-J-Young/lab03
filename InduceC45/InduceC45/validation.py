import argparse
import math
import pandas as pd
import numpy as np
from anytree import Node, RenderTree
import subprocess
import os
import ast
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('training_file',
                    type = argparse.FileType('r'))

parser.add_argument('--restrictionsFile',
                    type = argparse.FileType('r'))

parser.add_argument('n', type=int)
args = parser.parse_args()

trainingSet = pd.read_csv(args.training_file.name, skiprows =[1,2])

slices_num = args.n
if (slices_num == -1):
    slices_num = len(trainingSet.index)-1

#slice dataset into n slices of equal size
shuffled = trainingSet.sample(frac=1)
slices = np.array_split(shuffled, slices_num)

overall_confusion_matrix = {}

total_classified = 0
total_correct = 0
total_incorrect = 0
total_average_accurate_rate = 0
total_average_error_rate = 0

def get_rates(c_mat):
    correct_count = 0
    incorrect_count = 0
    for actual in c_mat:
        for key, value in c_mat[actual].items():
            if (key == actual):
                correct_count += value
            else:
                incorrect_count += value
    total = correct_count + incorrect_count
    return (correct_count, incorrect_count)

def sum_mat(overall_confusion_matrix, c_mat):
    print("xd")
    if (not overall_confusion_matrix): #if overall_confusion_matrix is empty set it to c_mat
        print("yes")
        overall_confusion_matrix = c_mat.copy()
    else: #else loop through c_mat and accumulate values
        for actual in c_mat:
            for key, value in c_mat[actual].items():
                overall_confusion_matrix[actual][key] += value
    return overall_confusion_matrix


for idx, slice in enumerate(slices): #each slice must be designated as a holdout set once
    tmp_slices = slices[0:idx] + slices[idx+1:]
    training_set = pd.concat(tmp_slices)
    
    train_file = open(os.path.splitext(args.training_file.name)[0] + "-train" + ".csv", "w")
    train_file.write(training_set.to_csv())
    train_file.close()

    train_file = open(os.path.splitext(args.training_file.name)[0] + "-validate" + ".csv", "w")
    train_file.write(slice.to_csv())
    train_file.close()

    if (args.restrictionsFile):
        c45 = subprocess.call("python3 InduceC45.py " + os.path.splitext(args.training_file.name)[0] + "-train" + ".csv" + " " + args.restrictionsFile.name)
        c45.wait()
        classify = subprocess.run("python3 classify.py " + os.path.splitext(args.training_file.name)[0] + "-validate" + ".csv" + " " + os.path.splitext(args.training_file.name)[0] + ".json ", check=True, stdout=subprocess.PIPE, universal_newlines=True)
        c_output = classify.stdout
        lines = c_output.splitlines()
        #get keys in input c_mat, set overall 
        #for each key in the input c_mat, check if that key exists in the overall c_mat
        inv_c_mat = ast.literal_eval(lines[6]) #indivial confusion matrix
        #calculate average accuracy and error rate for this individual c_mat
        #mats.append(ast.literal_eval(lines[6]))
        overall_confusion_matrix = sum_mat(overall_confusion_matrix, inv_c_mat)
        print(overall_confusion_matrix)
        counts = get_rates(inv_c_mat) #returns correct an incorrect counts
        total = counts[0] + counts[1]
        total_classified += total
        total_correct += counts[0]
        total_incorrect += counts[1]
        total_average_accurate_rate += float(counts[0])/total
        total_average_error_rate += float(counts[1])/total
    else:
        c45 = subprocess.run("python3 InduceC45.py " + os.path.splitext(args.training_file.name)[0] + "-train" + ".csv")
        #print("python3 classify.py " + args.training_file.name + " " + os.path.splitext(args.training_file.name)[0] + ".json ")
        classify = subprocess.run("python3 classify.py " + os.path.splitext(args.training_file.name)[0] + "-validate" + ".csv" + " " + os.path.splitext(args.training_file.name)[0] + ".json ", check=True, stdout=subprocess.PIPE, universal_newlines=True)
        c_output = classify.stdout
        lines = c_output.splitlines()
        #get keys in input c_mat, set overall 
        #for each key in the input c_mat, check if that key exists in the overall c_mat
        inv_c_mat = ast.literal_eval(lines[6]) #indivial confusion matrix
        #calculate average accuracy and error rate for this individual c_mat
        #mats.append(ast.literal_eval(lines[6]))
        overall_confusion_matrix = sum_mat(overall_confusion_matrix, inv_c_mat)
        print(overall_confusion_matrix)
        counts = get_rates(inv_c_mat) #returns correct an incorrect counts
        total = counts[0] + counts[1]
        total_classified += total
        total_correct += counts[0]
        total_incorrect += counts[1]
        total_average_accurate_rate += float(counts[0])/total
        total_average_error_rate += float(counts[1])/total


print("Overall confusion matrix: " + str(overall_confusion_matrix))
print("Overall accuracy: " + str(total_correct/total_classified))
print("average accuracy: " + str(total_average_accurate_rate/slices_num))
print("Overall error: " + str(total_incorrect/total_classified))
print("average error: " + str(total_average_error_rate/slices_num))
            


        