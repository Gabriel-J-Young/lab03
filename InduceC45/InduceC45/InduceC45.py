import argparse
import math
import pandas as pd
from anytree import Node, RenderTree

parser = argparse.ArgumentParser()
parser.add_argument('TrainingSetFile',
                    type = argparse.FileType('r'))

parser.add_argument('--restrictionsFile',
                    type = argparse.FileType('r'))
args = parser.parse_args()

trainingSet = pd.read_csv(args.TrainingSetFile.name, skiprows =[1,2])

print(trainingSet)

#D: training dataset (pandas dataframe)
#A: list of attributes
#thresh: threshhold for splitting
#returns Tree root
def C45(D, A, classifier, thresh):
    print("iloc?", D.iloc[:, -1])
    print("attributes", D.iloc[:, -1].mode()[0])
    print("classifiers", D[classifier])
    print("A", A)
    #check termination conditions
    #first: if all of the dataset's attributes have the same class label c, if so, create 
    #a tree with a single node and assign class label c
    if (all_same(D[classifier])): #look at last column (should be classes)
        leaf = Node(D.iloc[-1, -1]) #look at last row and col & set leaf node to found class
        T = leaf
    elif (len(A) == 0): #if we have no more attributes to look at...
        c = D.iloc[:, -1].mode()[0] #mode returns array lol
        leaf = Node(c) #set leaf node to plurality class
        T = leaf
    else:
        best_A = select_splitting_attribute(A, D, thresh) #best_A is the best attribute
        print("BEST ATTRIBUTE:", best_A)
        if (best_A) == None: #no attribute is good enough for a split
            c = D.iloc[:, -1].mode()[0] #mode returns array lol
            leaf = Node(c) #set leaf node to plurality class
            T = leaf
        else: #construct non-leaf node
            T = Node(best_A) #this node should be labeled with an attribute
            for attr_inst in D[best_A].unique(): #for each instance of that attribute
                Dv = D[D[best_A]==attr_inst]
                if (not Dv.empty):
                    edge_node = Node(attr_inst)
                    print("t:", T)
                    print("best_A", best_A)
                    print(A)
                    new_A = A.copy()
                    new_A.remove(best_A)
                    branch = C45(Dv, new_A, classifier, thresh)
                    branch.parent = edge_node 
                    edge_node.parent = T
                    #T.children += (edge_node,) #append branch to T with edge labeled attr_inst
    return T

    
def select_splitting_attribute(A, D, thresh):
    gains = {}
    attrs = D.iloc[-1, -1:].unique()
    base_entropy = get_entropy(D)
    print("entropy:",base_entropy)
    #class_labels = D.iloc[:, -1:].unique()
    #for label in class_labels: #calculate entropy of D
       
    print("A", A)
    for attr in A: #for each attribute passed in...
        print("attr12", attr)
        attr_entropy = get_attr_entropy(attr, D)#calculate entropy of D after being split by attr
        print("entropy of", attr, attr_entropy)
        gains[attr] = base_entropy - attr_entropy
        print("entropy calc:", base_entropy,"-", attr_entropy)
        print("gains", gains)
    best_attr = max(gains, key=gains.get)#find highest info gain
    if (gains[best_attr] > thresh):
        return best_attr;
    return None;

def get_entropy(D):
    total_rows = len(D.index)
    entropy = 0
    #print(D.iloc[:, -1:].value_counts())
    for count in D.iloc[:, -1:].value_counts().iteritems():
        #print(count[0][0])
        entropy += (count[1]/total_rows) * math.log(count[1]/total_rows)
    return entropy * -1

def get_attr_entropy(attr, D):
    #D is a pandas dataframe, attr is an attribute of that dataframe (e.g. Color)
    #returns the entropy of D when D is split by that attr.
    total_rows = len(D.index)
    total_attr_entropy = 0;
    attr_instances = D[attr].unique()
    #print("attr instances:",D[attr].unique())
    for attr_instance in attr_instances:
        #print(" attr_i", attr_instance)
        D_matching_attr = D[D[attr]==attr_instance]
        #print(" count of attr:", len(D_matching_attr))
        attr_instance_entropy = get_entropy(D_matching_attr)

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

T = C45(trainingSet, A, classifier, .2)
print(RenderTree(T))

