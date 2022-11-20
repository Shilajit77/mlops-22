import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from joblib import dump
import pdb
from sklearn.metrics import classification_report
import argparse 
from sklearn.metrics import f1_score,accuracy_score
from sklearn import datasets, svm, metrics
from sklearn.tree import DecisionTreeClassifier


digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images
data = digits.images.reshape((n_samples, -1))

parser = argparse.ArgumentParser(description = 'Model with random state')
parser.add_argument('--clf_name', type = str)

parser.add_argument('--random_state', type = int)
args = parser.parse_args()

clf_name = args.clf_name
random_state = args.random_state

X_train1, X_test1, y_train1, y_test1 = train_test_split(data, digits.target, test_size=0.33, random_state=random_state)
model_set = {'svm': svm.SVC(gamma=0.01),'dtree': DecisionTreeClassifier()}

def train_with_clf_random_state(clf_name,random_state):
    clf = model_set[clf_name]
    clf.fit(X_train1,y_train1)
    ypred = clf.predict(X_test1)
    f1 = f1_score(y_test1.reshape(-1,1), ypred.reshape(-1,1), average='macro')
    print("test accuracy: ",accuracy_score(y_test1.reshape(-1,1), ypred.reshape(-1,1)))
    print("\ntest macro-f1: ",f1)

train_with_clf_random_state(clf_name,random_state)