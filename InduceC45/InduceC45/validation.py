import argparse
import math
import pandas as pd
import numpy as np
from anytree import Node, RenderTree
import subprocess
import os
import re

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

overall_confusion_matrix = [[]]

total_classified = 0
total_correct = 0
total_incorrect = 0
total_average_accurate_rate = 0
total_average_error_rate = 0

for idx, slice in enumerate(slices): #each slice must be designated as a holdout set once
    tmp_slices = slices[0:idx] + slices[idx+1:]
    training_set = pd.concat(tmp_slices)
    training_set.to_csv()
    if (args.restrictionsFile):
        c45 = subprocess.call("InduceC45.py " + args.training_file.name + " " + args.restrictionsFile.name)
        c45.wait()
        classify = subprocess.run("classify.py " + args.training_file.name + " example_nursery.json ", check=True, stdout=subprocess.PIPE, universal_newlines=True)
    else:
        c45 = subprocess.run("python3 InduceC45.py " + args.training_file.name)
        print("python3 classify.py " + args.training_file.name + " " + os.path.splitext(args.training_file.name)[0] + ".json ")
        classify = subprocess.run("python3 classify.py " + args.training_file.name + " " + os.path.splitext(args.training_file.name)[0] + ".json ", check=True, stdout=subprocess.PIPE, universal_newlines=True)
        c_output = classify.stdout
        lines = c_output.splitlines()
        total_classified += int(''.join(filter(str.isdigit, lines[0])))
        total_correct += int(''.join(filter(str.isdigit, lines[1])))
        total_incorrect += int(''.join(filter(str.isdigit, lines[2])))
        total_average_accurate_rate += int(''.join(filter(str.isdigit, lines[3])))
        total_average_error_rate += int(''.join(filter(str.isdigit, lines[4])))

print("Overall confusion matrix: ")
print("Overall accuracy: " + str(total_correct/total_classified))
print("average accuracy: " + str(total_average_accurate_rate/slices_num))
print("Overall error: " + str(total_incorrect/total_classified))
print("average error: " + str(total_average_error_rate/slices_num))
            


        