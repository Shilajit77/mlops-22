import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from joblib import dump
import pdb
from sklearn.metrics import classification_report


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#resize(image, (100, 100)).shape(100, 100)

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def get_hparams(X_train,y_train,X_test,y_test):
    gamma_list = [0.01, 0.005, 0.001]
    c_list = [0.1, 0.2, 0.5]
    best_acc=-1
    for i in gamma_list:
        for j in c_list:
            model = svm.SVC(gamma=i,C=j)
            model = model.fit(X_train,y_train)
            acc = model.score(X_test,y_test)
            if(acc>best_acc):
                best_acc = acc
                best_g = i
                best_c = j
                best_params = {'C':best_c, 'Gamma': best_g}
                print("Acc: "+str(best_acc)+" Gamma: ",str(best_g)+" C: ",str(best_c))
    print("\nFinal best parameters are: ")
    print(" Gamma: ",str(best_g)+" C: ",str(best_c))
    print("\n")
    return best_acc,best_g,best_c,model,best_params
    
    
def predict(model,x_test,y_test):
    ypred = model.predict(x_test)
    print(classification_report(y_test,ypred))
    return ypred














train_frac = 0.7
test_frac = 0.2
dev_frac = 0.1
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images
data = digits.images.reshape((n_samples, -1))
dev_test_frac = 1-train_frac


X_train1, X_test1, y_train1, y_test1 = train_test_split(data, digits.target, test_size=0.33, random_state=42)

X_train2, X_test2, y_train2, y_test2 = train_test_split(data, digits.target, test_size=0.33, random_state=42)

X_train3, X_test3, y_train3, y_test3 = train_test_split(data, digits.target, test_size=0.33, random_state=10)


#best_acc,best_g,best_c,model,best_params = get_hparams(X_train1,y_train1,X_test1,y_test1)
#ypred = predict(model,X_test,y_test)
#print(ypred)

def test_if_same():
    assert X_train1.all() == X_train2.all()
    assert X_test1.all() == X_test2.all()
def test_if_not_same():
    assert (X_test1 != X_test3).any()
    assert (X_train2 != X_train3).any()