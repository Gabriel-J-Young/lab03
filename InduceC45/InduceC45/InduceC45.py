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

trainingSet = pd.read_csv(args.TrainingSetFile.name)

print(trainingSet)

#D: training dataset (pandas dataframe)
#A: list of attributes
#thresh: threshhold for splitting
#returns Tree root
def C45(D, A, thresh):
    print(D.iloc[:, -1].mode()[0])
    #check termination conditions
    #first: if all of the dataset's attributes have the same class label c, if so, create 
    #a tree with a single node and assign class label c
    if (all_same(D.iloc[:, -1:])): #look at last column (should be classes)
        leaf = Node(D.iloc[-1, -1]) #look at last row and col & set leaf node to found class
        T = leaf
    elif (len(A) == 0): #if we have no more attributes to look at...
        c = D.iloc[:, -1].mode()[0] #mode returns array lol
        leaf = Node(c) #set leaf node to plurality class
        T = leaf
    else:
        Ag = selectSplittingAttribute(A, D, 1) #Ag is the best attribute
        if (Ag) == None: #no attribute is good enough for a split
            c = D.iloc[:, -1].mode()[0] #mode returns array lol
            leaf = Node(c) #set leaf node to plurality class
            T = leaf
        else: #construct non-leaf node
            T = Node(Ag) #this node should be labeled with an attribute
            for v in D[Ag].unique(): #for v in dom(Ag) v is attribute val
                Dv = D.query('@Ag == @v') #possible missuse of query
                if (not Dv.empty):
                    edge_node = Node(v)
                    branch = C45(Dv, A.remove(Ag), thresh)
                    branch.parent = edge_node
                    edge_node.parent = T
                    branch.parent(T) #append branch to T with edge labeled v
    return T

    
def selectSplittingAttribute(A, D, thresh):
    gain = []
    attrs = D.iloc[-1, -1:].unique()
    entropy = getEntropy(D)
    print(entropy)
    #class_labels = D.iloc[:, -1:].unique()
    #for label in class_labels: #calculate entropy of D
       

    for attr in A: #for each attribute passed in...
        get_attr_entropy(attr, D)#calculate entropy of D after being split by attr
        #calculate info gain
        #print("spacer")
    #best = max(gain) #find highest info gain
    #if (best > thresh):
    #    return best;
    #return None;

def getEntropy(D):
    total_rows = len(D.index)
    entropy = 0
    #print(D.iloc[:, -1:].value_counts())
    for count in D.iloc[:, -1:].value_counts().iteritems():
        
        print(count[0][0])
        entropy += (count[1]/total_rows) * math.log(count[1]/total_rows)
    return entropy * -1

def get_attr_entropy(attr, D):
    total_rows = len(D.index)
    for attr_val_count in D[attr].value_counts().iteritems():
        print(attr)
        print("attr", attr_val_count[0])
        print("type", type(D))
        print("color", D['Color'])
        print(D[D[attr]==attr_val_count[0]])
        attr_val_count[1]/total_rows * getEntropy(D.query('@attr == @attr_val_count[0]'))




#test to see if all values in col are unique
def all_same(s):
    a = s.to_numpy()
    return (a[0] == a).all()

A = trainingSet.columns.values
print(trainingSet.columns.values)
T = C45(trainingSet, A, .5)
print(RenderTree(T))

