import argparse
import math
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('TrainingSetFile',
                    type = argparse.FileType('r'))

parser.add_argument('restrictionsFile',
                    nargs='?',
                    type = argparse.FileType('r'))
args = parser.parse_args()

trainingSet = pd.read_csv(args.TrainingSetFile.name)

class_label_index = -1
class_label_name = trainingSet.columns.values[class_label_index]
print(class_label_name)

#returns num dist (non normalized), cata dist (normalized)
def get_distance(d, x, numeric, m):
    sum_squared_dist = 0
    cata_match = 0
    #need to change to not use class label
    for key, value in numeric.items():
        if (not key == class_label_name):
            if (value == 0):
                #print("numeric")
                sum_squared_dist += (d[1][key] - x[1][key])**2
            else:
                #print("cata")
                if (d[1][key] == x[1][key]):
                    cata_match += 1
   
    return (cata_match/m, math.sqrt(sum_squared_dist))

def knn(D, k, x, numeric, num_numeric, m):
    #D - dataset
    #num_numeric - number of nearest neighbors
    #x - point to classify
    #numeric - what attributes are numeric
    #k - number of numeric varibles
    #m - number of catagorical variables
    neighbors = [()] *(k+1) #leave one space for x
    dists = []

    for d in D.iterrows():   # compute distances
        dists.append((d[1][class_label_name], get_distance(d, x, numeric, m)))
    max_dist = 0
    for d in dists:
        if (d[1][1] > max_dist):
            max_dist = d[1][1]
    dists_norm = [] #filled with tuples, item, and distance
    for d in dists:
        #print("num: " + str(d[1][1]/max_dist))
        #print("cata: " + str((d[1][0])))
        dists_norm.append((d[0], (d[1][1]/max_dist)*(num_numeric/(m+num_numeric) + (d[1][0])*(m/(m+num_numeric)))))

    #print(x[1])
    dists_norm.sort(key=lambda x: x[1])
    

    k_nearest = []
    for d in dists_norm[1:k+1]:
        k_nearest.append(d[0])

    #print(k_nearest)

    return max(set(k_nearest), key=k_nearest.count)
        



numeric = {} #list of numberic attributes
num_numeric = 0
for col in trainingSet:
    numeric[col] = trainingSet[col].iloc[0]
    if (trainingSet[col].iloc[0] == 0):
        num_numeric += 1
#print(numeric)

D = trainingSet.iloc[:][2:]

class_labels = []
for idx, row in enumerate(D.iterrows()):
    class_labels.append((row[1][class_label_index], knn(D, 5, row, numeric, num_numeric, len(numeric)-num_numeric))) 

print(class_labels)

