import argparse
import math
import pandas as pd
import numpy as np
import time

parser = argparse.ArgumentParser()

parser.add_argument('TrainingSetFile',
                    type = argparse.FileType('r'))

parser.add_argument('k',
                    type = int)

parser.add_argument('restrictionsFile',
                    nargs='?',
                    type = argparse.FileType('r'))

args = parser.parse_args()

trainingSet = pd.read_csv(args.TrainingSetFile.name)

class_label_index = 0
class_label_name = trainingSet.columns.values[class_label_index]

numeric_names = [] #list of numberic attributes
cata_names = []
for col in trainingSet:
    if (not col == class_label_name):
        if (trainingSet[col].iloc[0] == 0 or trainingSet[col].iloc[0] == "0"):
            numeric_names.append(col)
        else:
            cata_names.append(col)

print(numeric_names)
print(class_label_name)

def distances(df, x):
 # df is a data frame
 # x is a vector

  data = np.array(df)

  return np.sqrt(np.sum(((data - x)**2), axis =1))


def knn(D, k, x, num_D):
    #D - dataset
    #k - number of nearest neighbors
    #x - point to classify
    #numeric - what attributes are numeric
    neighbors = [()] *(k+1) #leave one space for x
    #t = time.time()
    x_numeric_tmp = []
    for name in numeric_names:
        x_numeric_tmp.append(x[1][name])
    x_numeric = np.array(x_numeric_tmp)


    dists_numeric = distances(num_D, x_numeric)
    #print("np: " + str(time.time()-t))

    dists_cata = []
    class_labels = []

    #print(D[numeric_names].values.astype(float))
    #t = time.time()
    #for d in num_D: #compute numerical dists
    #    sum_squared_dist = 0
    #    for attr_idx, attr_val in enumerate(numeric_names):
    #        sum_squared_dist += (d[attr_idx] - float(x[1][attr_val]))**2
    #    dists_numeric.append(math.sqrt(sum_squared_dist))
    #print("num loop: " + str(time.time()-t))

    if (len(cata_names) != 0):
        for d in D[cata_names].iterrows(): #compute cata dists
            sum_cata_dist = 0
            for attr_val in cata_names:
                sum_cata_dist += d[1][attr_val] == x[1][attr_val]  
            dists_cata.append(sum_cata_dist)

    max_dist = np.array(dists_numeric).max()

    dists_norm = [] #filled with tuples, item, and distance
    if (len(cata_names) == 0):
        t = time.time()
        for d_num, label in zip(dists_numeric, D[class_label_name].tolist()):
            c_n = len(numeric_names)
            dists_norm.append((label,(d_num/max_dist)))
        #print("zip loop: " + str(time.time()-t))
    else:
        for d_num, d_cata, label in zip(dists_numeric, dists_cata, D[class_label_name].tolist()):
            c_m = len(cata_names)
            c_n = len(numeric_names)
            dists_norm.append((label,(d_num/max_dist)*(c_n/(c_m+d_num) + (d_cata)*(c_m/(c_m+c_n)))))

    #print(dists_norm)
    tmp_min = min(dists_norm, key = lambda t: t[1]) #remove ele with smallest dist
    dists_norm.remove(tmp_min)

    k_nearest = []
    for d in range(k):
        tmp_min = min(dists_norm, key = lambda t: t[1]) #remove ele with smallest dist
        dists_norm.remove(tmp_min)
        k_nearest.append(tmp_min[0])

    return max(set(k_nearest), key=k_nearest.count)

def print_c_mat(mat):
    print(",", end="")
    for key, val in mat.items():
        print(key, end=",")
    print()
    for key, val in mat.items():
        print(key, end=",")
        for key_in, val_in in val.items():
            print(val_in, end=",")
        print()
      
def print_stats(confusion_list, trainingSet):
    total_classified = len(confusion_list)
    total_classified_correct = 0
    total_classified_incorrect = 0
    confusion_lists = {} #each 
    for val in trainingSet.iloc[:,class_label_index].unique(): #for each possible class label
        confusion_lists[val] = {}
    for a_val in confusion_lists: # for each possible actual value
        for val in trainingSet.iloc[:,class_label_index].unique():
            confusion_lists[a_val][val] = 0

    for pair in confusion_list:
        confusion_lists[pair[1]][pair[0]] += 1
        #(predicted, actual)
        if (pair[0] == pair[1]):
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
    #attributes = confusion_lists.keys()
    #print(attributes)

    print_c_mat(confusion_lists)

D = trainingSet.iloc[:][2:]

for num in numeric_names: #drop all rows with "?"
    D = D[(D != "?").all(axis=1)]

class_labels = []
num_D = D[numeric_names].values.astype(float)
for idx, row in enumerate(D.iterrows()):
    class_labels.append((knn(D, args.k, row, num_D), row[1][class_label_name]))
    #print(idx)

print_stats(class_labels, D)