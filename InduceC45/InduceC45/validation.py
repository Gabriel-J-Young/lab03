import argparse
import math
import pandas as pd
from anytree import Node, RenderTree

parser = argparse.ArgumentParser()
parser.add_argument('training_file',
                    type = argparse.FileType('r'))

parser.add_argument('--restrictionsFile',
                    type = argparse.FileType('r'))

parser.add_argument('n', type=int)
args = parser.parse_args()

