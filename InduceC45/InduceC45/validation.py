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

slices_num = args.n
if (slices_num == -1):
    slices_num = len(trainingSet.index)-1

#slice dataset into n slices of equal size
shuffled = trainingSet.sample(frac=1)
slices = np.array_split(shuffled, slices_num)  

for idx, slice in enumerate(slices): #each slice must be designated as a holdout set once
    tmp_slices = slices[0:idx] + slices[idx+1:]
    training_set = pd.concat(tmp_slices)
    training_set.to_csv()
    if (args.restrictionsFile):
        c45 = subprocess.call("InduceC45.py " + args.training_file.name + " " + args.restrictionsFile.name)
        c45.wait()
        classify = subprocess.call("classify.py " + args.training_file.name + " example_nursery.json " + n,  check=True, stdout=subprocess.PIPE)
    else:
        c45 = subprocess.call("python3 InduceC45.py " + args.training_file.name)
        c45.wait()
        classify = subprocess.call("classify.py " + args.training_file.name + " example_nursery.json " + n)


        