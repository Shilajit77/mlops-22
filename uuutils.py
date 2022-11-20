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
