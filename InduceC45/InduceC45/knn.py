import argparse
import math
import pandas as pd

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
    
    
    
numeric = {} #list of numberic attributes
num_numeric = 0
for col in trainingSet:
    numeric[col] = trainingSet[col].iloc[0]
    if (trainingSet[col].iloc[0] == 0):
        num_numeric += 1

D = trainingSet.iloc[:][2:]

class_labels = []
for idx, row in enumerate(D.iterrows()):
    class_labels.append((knn(D, args.k, row, numeric, num_numeric, len(numeric)-num_numeric), row[1][class_label_index])) 

print_stats(class_labels, D)




