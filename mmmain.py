import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from joblib import dump
import pdb
from sklearn.metrics import classification_report
#import argparse
from sklearn.metrics import f1_score,accuracy_score

from uuutils import (
    get_hparams,
    predict,
)










train_frac = 0.7
test_frac = 0.2
dev_frac = 0.1
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images
data = digits.images.reshape((n_samples, -1))
dev_test_frac = 1-train_frac

seed = 42

X_train1, X_test1, y_train1, y_test1 = train_test_split(data, digits.target, test_size=0.33, random_state=seed)


best_acc,best_g,best_c,model,best_params = get_hparams(X_train1,y_train1,X_test1,y_test1)
ypred = predict(model,X_test1,y_test1)
#print(ypred)
f1 = f1_score(y_test1.reshape(-1,1), ypred.reshape(-1,1), average='macro')
print("test accuracy: ",accuracy_score(y_test1.reshape(-1,1), ypred.reshape(-1,1)))
print("\ntest macro-f1: ",f1)


d = {'test accuracy: ':accuracy_score(y_test1.reshape(-1,1), ypred.reshape(-1,1)),'test macro-f1:':f1}

#with open('results/svm_42.txt', 'w') as f:
   # f.write('test accuracy: ',str(f1))
   # f.write('test macro-f1: ",str(accuracy_score(y_test1.reshape(-1,1), ypred.reshape(-1,1))))
dump(model,"models/"+"Svm" + "_" + str(best_params) +"Random_state: "+str(seed)+ ".joblib")
