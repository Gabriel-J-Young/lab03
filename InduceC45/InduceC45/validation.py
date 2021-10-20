import argparse
import math
import pandas as pd
import numpy as np
from anytree import Node, RenderTree
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('training_file',
                    type = argparse.FileType('r'))

parser.add_argument('--restrictionsFile',
                    type = argparse.FileType('r'))

parser.add_argument('n', type=int)
args = parser.parse_args()

trainingSet = pd.read_csv(args.training_file.name, skiprows =[1,2])

#slice dataset into n slices of equal size
shuffled = trainingSet.sample(frac=1)
slices = np.array_split(shuffled, args.n)  

for slice in slices: #each slice must be designated as a holdout set once
    tmp_slices = slices.copy()
    tmp_slices.remove(slice)
    training_set = pd.concat(tmp_slices)
    training_set.to_csv()
    if (args.restrictionsFile):
        subprocess.Popen("InduceC45.py " + args.training_file.name + " " + args.restrictionsFile.name)
    else:
        subprocess.Popen("python3 InduceC45.py " + args.training_file.name)
    print(part,'\n')