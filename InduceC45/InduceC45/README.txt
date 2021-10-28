Xander Wallace
xwallace@calpoly.edu
Gabriel Young
gjyoung@calpoly.edu

On line 22 in InduceC45.py and line 112 in randomForest, the skiprows argument may need to be modified if the dataset has less rows with metadata. 
In knn.py, line 23 the index of the column with class labels must be set.
On line 12 in InduceC45.py the gain_threshold can be modified.

running c45:
 python3 InduceC45.py <TrainingSetFile.csv> [<restrictionsFile>]
 python3 validation.py <TrainingSetFile.csv>, number folds, [<restrictionsFile>]

running randomForest: 
 randomForest.py num_attributes num_datapoints num_trees

running KNN:
 knn.py <TrainingSetFile.csv> k
 