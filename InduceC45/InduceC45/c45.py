import math
import pandas
import json
import numpy as np
from anytree import Node, RenderTree
from six import assertRaisesRegex         

continuous_threshold = 10


def C45(D, thresh):
    #wrapper for C45 algorithm. Setup attribute list, classifier, and continuous list. Then call C45_lp
    A = []
    for attribute in D.columns.values:
        A.append(attribute)
    classifier = A[-1]
    del A[-1]
    cont = get_continuous(D, A)
    return C45_lp(D, A, classifier, cont, thresh)

def get_continuous(D, A):
    #returns a dict telling continuous attributes in A
    cont = {}
    for attr in A:
        try:
            i = int(D[attr].iloc[0])
            if len(D[attr].unique()) > continuous_threshold:
                cont[attr] = True
            else:
                cont[attr] = False
        except ValueError:
            cont[attr] = False
    return cont


def C45_lp(D, A, classifier, cont, thresh):
    #D: training dataset (pandas dataframe)
    #A: list of attributes
    #thresh: threshhold for splitting
    #returns Tree root
    #check termination conditions
    #first: if all of the dataset's attributes have the same class label c, if so, create 
    #a tree with a single node and assign class label c
    if (all_same(D[classifier])): #look at last column (should be classes)
        T = Node(D[classifier].iloc[-1]) #look at last row and col & set leaf node to found class
        T.purity = float(1.0);
    elif (len(A) == 0): #if we have no more attributes to look at...      
        T = make_plurality_leaf(D, classifier)
    else:
        ssa = select_splitting_attribute(A, D, classifier, cont, thresh) #best_A is the best attribute
        if ssa == None: #no attribute is good enough for a split
            T = make_plurality_leaf(D,classifier)
        else: #construct non-leaf node
            best_A = ssa[0]
            best_split = ssa[1]
            T = Node(str(best_A)) #this node should be labeled with an attribute
            if cont[best_A]:
                #The lack of sanitization here will cause a bug if input data includes > or < characters
                edge_node_g = Node(">="+str(best_split))
                new_A_g = A.copy()
                D_greater = D[D[best_A] >= best_split]
                branch_g = C45_lp(D_greater, new_A_g, classifier,cont, thresh)
                branch_g.parent = edge_node_g
                edge_node_g.parent = T

                edge_node_l = Node("<"+str(best_split))
                new_A_l = A.copy()
                D_less = D[D[best_A] < best_split]
                branch_l = C45_lp(D_less, new_A_l, classifier,cont, thresh)
                branch_l.parent = edge_node_l
                edge_node_l.parent = T
            else:
                for attr_inst in D[best_A].unique(): #for each instance of that attribute
                    Dv = D[D[best_A]==attr_inst]
                    if (not Dv.empty):
                        edge_node = Node(str(attr_inst))
                        new_A = A.copy()
                        new_A.remove(best_A)
                        branch = C45_lp(Dv, new_A, classifier,cont, thresh)
                        branch.parent = edge_node 
                        edge_node.parent = T
    return T

def make_plurality_leaf(D, classifier):
    #set leaf node to plurality class and return that node
    c = D[classifier].mode()[0] #mode returns array lol
    count_c = len(D[D[classifier]==c])
    
    leaf = Node(str(c)) #set leaf node to plurality class
    leaf.purity = float(count_c/len(D[classifier]))
    return leaf
    
def select_splitting_attribute(A, D, classifier, cont, thresh):
    gains = {}
    best_splits = {}
    base_entropy = get_entropy(D, classifier, len(D.index))
    #for label in class_labels: #calculate entropy of D
    for attr in A: #for each attribute passed in...
        if cont[attr]: #A is continuous
            fbs = find_best_split(attr, D, classifier)
            best_splits[attr] = fbs[0]
            attr_entropy = fbs[1]
            #suboptimal
        else:
            best_splits[attr] = -1 # dummy value, never used
            attr_entropy = get_attr_entropy(attr, D, classifier)#calculate entropy of D after being split by attr
        gains[attr] = base_entropy - attr_entropy
    best_attr = max(gains, key=gains.get)#find highest info gain
    if (gains[best_attr] > thresh):
        return (best_attr, best_splits[best_attr])
    return None;

def get_entropy(D, classifier, total_rows):
    weights = np.array([count[1]/total_rows for count in D[classifier].value_counts().iteritems()])
    entropy = np.sum(weights*np.log2(weights))
    return entropy*(-1)

def find_best_split(attr, D, classifier):
    splittable_points = D[attr].unique()
    split_entropies = {}
    total_rows = len(D.index)
    for split in splittable_points:
        D_greater = D[D[attr] >= split]
        gtr_rows = len(D_greater.index)
        D_less = D[D[attr] < split]
        less_rows= len(D_less.index)
        split_entropies[split] = (less_rows/total_rows)*get_entropy(D_less, classifier, less_rows) + \
            (gtr_rows/total_rows)*get_entropy(D_greater, classifier, gtr_rows)
    minimum_split = min(split_entropies, key = split_entropies.get)
    return (minimum_split, split_entropies[minimum_split])

def get_attr_entropy(attr, D, classifier):
    #D is a pandas dataframe, attr is an attribute of that dataframe (e.g. Color)
    #returns the entropy of D when D is split by that attr.
    total_rows = len(D.index)
    total_attr_entropy = 0;
    attr_instances = D[attr].unique()
    for attr_instance in attr_instances:
        D_matching_attr = D[D[attr]==attr_instance]
        attr_instance_entropy = get_entropy(D_matching_attr, classifier, total_rows)
        total_attr_entropy += len(D_matching_attr)/total_rows * attr_instance_entropy
    return total_attr_entropy


#test to see if all values in col are unique
def all_same(s):
    a = s.to_numpy()
    return (a[0] == a).all()

#
#------------------JSON SECTION ---------------------------
#

#T: tree to tranlate to json
def tree_to_json_str(T, name):
    #TODO: add continuous functionality
    #given a c45 tree, returns a json
    data = {}
    data['dataset'] = name
    data['node'] = get_node(T)

    return json.dumps(data, indent=4)

def tree_to_json(T, name):
    #TODO: add continuous functionality
    #given a c45 tree, returns a json
    data = {}
    data['dataset'] = name
    data['node'] = get_node(T)

    return data

def get_node(T):
    data = {}
    data['var'] = T.name
    data['edges'] = get_edges(T.children)
    return data

def get_edges(children):
    edges = []
    for child in children:
        data = {}
        data['value'] = child.name
        if (hasattr(child.children[0],'purity')): #must be leaf
            data['leaf'] = {"decision": str(child.children[0].name),"p": child.children[0].purity}
        else: #must be node
            data['node'] = get_node(child.children[0])
        name_dict = {}
        name_dict['edge'] = data
        edges.append(name_dict)
    return edges

def restrict_dataset(D,A, restrict_list):
    for i in range(0,len(restrict_list)):
        if int(restrict_list[i])==1:
            D = D.drop(columns = [A[i]])
            del A[i]
    return D