import argparse
import math
import pandas as pd
from anytree import Node, RenderTree
from six import assertRaisesRegex

parser = argparse.ArgumentParser()
parser.add_argument('TrainingSetFile',
                    type = argparse.FileType('r'))

parser.add_argument('--restrictionsFile',
                    type = argparse.FileType('r'))
args = parser.parse_args()

trainingSet = pd.read_csv(args.TrainingSetFile.name)

print(trainingSet)

#D: training dataset (pandas dataframe)
#A: list of attributes
#thresh: threshhold for splitting
#returns Tree root
def C45(D, A, classifier, thresh):
    print()
    print("------ running with A=", A,"------")
    print(D)
    #check termination conditions
    #first: if all of the dataset's attributes have the same class label c, if so, create 
    #a tree with a single node and assign class label c
    if (all_same(D[classifier])): #look at last column (should be classes)
        T = Node(D[classifier].iloc[-1]) #look at last row and col & set leaf node to found class
        T.purity = 1.0;
    elif (len(A) == 0): #if we have no more attributes to look at...      
        T = make_plurality_leaf(D, classifier)
    else:
        best_A = select_splitting_attribute(A, D, thresh, classifier) #best_A is the best attribute
        print("BEST ATTRIBUTE:", best_A)
        if (best_A) == None: #no attribute is good enough for a split
            T = make_plurality_leaf(D,classifier)
        else: #construct non-leaf node
            T = Node(best_A) #this node should be labeled with an attribute
            for attr_inst in D[best_A].unique(): #for each instance of that attribute
                Dv = D[D[best_A]==attr_inst]
                if (not Dv.empty):
                    edge_node = Node(attr_inst)
                    new_A = A.copy()
                    new_A.remove(best_A)
                    branch = C45(Dv, new_A, classifier, thresh)
                    branch.parent = edge_node 
                    edge_node.parent = T
    return T

def make_plurality_leaf(D, classifier):
    #set leaf node to plurality class and return that node
    c = D[classifier].mode()[0] #mode returns array lol
    print("------- Mode", c)
    leaf = Node(c) #set leaf node to plurality class
    leaf.purity = D[classifier][c]/len(D[classifier])
    return leaf
    
def select_splitting_attribute(A, D, thresh, classifier):
    gains = {}
    attrs = D.iloc[-1, -1:].unique()
    base_entropy = get_entropy(D, classifier)
    #class_labels = D.iloc[:, -1:].unique()
    #for label in class_labels: #calculate entropy of D

    for attr in A: #for each attribute passed in...
        attr_entropy = get_attr_entropy(attr, D)#calculate entropy of D after being split by attr
        #print("entropy of", attr, attr_entropy)
        gains[attr] = base_entropy - attr_entropy
    best_attr = max(gains, key=gains.get)#find highest info gain
    if (gains[best_attr] > thresh):
        return best_attr;
    return None;

def get_entropy(D, classifier):
    total_rows = len(D.index)
    entropy = 0
    for count in D[classifier].value_counts().iteritems():
        entropy -= (count[1]/total_rows) * math.log(count[1]/total_rows, 2)
    return entropy

def get_attr_entropy(attr, D):
    #D is a pandas dataframe, attr is an attribute of that dataframe (e.g. Color)
    #returns the entropy of D when D is split by that attr.
    total_rows = len(D.index)
    total_attr_entropy = 0;
    attr_instances = D[attr].unique()
    for attr_instance in attr_instances:
        D_matching_attr = D[D[attr]==attr_instance]
        attr_instance_entropy = get_entropy(D_matching_attr, classifier)
        total_attr_entropy += len(D_matching_attr)/total_rows * attr_instance_entropy
    return total_attr_entropy


#test to see if all values in col are unique
def all_same(s):
    a = s.to_numpy()
    return (a[0] == a).all()

A = []
for attribute in trainingSet.columns.values:
    A.append(attribute)
classifier = A[-1]
del A[-1]

T = C45(trainingSet, A, classifier, 0.0)
print(RenderTree(T))

