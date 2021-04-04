'''
make_cv_folds.py

Creates the train/test index files
for 10-fold cross validation for the
10fold_idx directory, required by
the input data format of DGCNN.

nosajmik, April 2021
'''

import os
import sys
import sklearn.model_selection
import numpy as np

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <DGCNN datafile>")
    exit(1)

with open(sys.argv[1], 'r') as f:
    num_samples = int(f.readline().strip())

kf = sklearn.model_selection.KFold(n_splits=10, 
                    shuffle=True, random_state=388)

os.makedirs("10fold_idx")
foldnum = 1

for train_idx, test_idx in kf.split(np.zeros(num_samples)):
    print(f"Fold {foldnum}, train: {train_idx}")
    print(f"Fold {foldnum}, test: {test_idx}")

    with open(f"10fold_idx/train_idx-{foldnum}.txt", 'a') as f:
        for elt in train_idx:
            f.write(str(elt) + '\n')
    
    with open(f"10fold_idx/test_idx-{foldnum}.txt", 'a') as f:
        for elt in test_idx:
            f.write(str(elt) + '\n')

    foldnum += 1