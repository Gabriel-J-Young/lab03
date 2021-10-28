Xander Wallace
xwallace@calpoly.edu
Gabriel Young
gjyoung@calpoly.edu

On line 21, the skiprows argument may need to be modified if the
dataset has less rows with metadata. 

running c45:
 python3 InduceC45.py <TrainingSetFile.csv> [<restrictionsFile>]
 python3 validation.py <TrainingSetFile.csv>, number folds, [<restrictionsFile>]

running randomForest: 
 randomForest.py num_attributes num_datapoints num_trees

running knn:
 