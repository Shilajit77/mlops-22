import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#resize(image, (100, 100)).shape(100, 100)

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# 1. set the ranges of hyper parameters 
gamma_list = [0.01, 0.005, 0.001]
c_list = [0.1, 0.2, 0.5] 

h_param_comb = [{'gamma':g, 'C':c} for g in gamma_list for c in c_list]

assert len(h_param_comb) == len(gamma_list)*len(c_list)


report = pd.DataFrame(h_param_comb)
#print(report.head())


train_frac = 0.7
test_frac = 0.2
dev_frac = 0.1

#PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()


n_samples = len(digits.images)
#digits = datasets.load_digits()
data = digits.images
data = digits.images.reshape((n_samples, -1))
dev_test_frac = 1-train_frac


best_acc = -1.0
best_model = None
best_h_params = None

#width, height = X_train.size

#print(width, height)


svm_accuracy = []

# 2. For every combination-of-hyper-parameter values
for i in range(0,5):
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=dev_test_frac, shuffle=True
    )
    X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
    )
    for cur_h_params in h_param_comb:

        #PART: Define the model
        # Create a classifier: a support vector classifier
        clf = svm.SVC()

        #PART: setting up hyperparameter
        hyper_params = cur_h_params
        clf.set_params(**hyper_params)


        #PART: Train model
        # 2.a train the model 
        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        # print(cur_h_params)
        #PART: get dev set predictions
        predicted_dev = clf.predict(X_test)

        # 2.b compute the accuracy on the validation set
        cur_acc = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_test)
        

        # 3. identify the combination-of-hyper-parameter for which validation set accuracy is the highest. 
        if cur_acc > best_acc:
            best_acc = cur_acc
            best_model = clf
            best_h_params = cur_h_params
            #print("Found new best acc with :"+str(cur_h_params))
            #print("New best val accuracy:" + str(cur_acc)
    svm_accuracy.append(cur_acc)
            
#predicted = best_model.predict(X_test)
#print("Best hyperparameters were:")
#print(cur_h_params)
#print(accuracy)
#accuracy = pd.DataFrame(accuracy,columns=['acc'])
#final_report = pd.concat([report,accuracy],axis=1)
#final_report
#print(svm_accuracy)





from sklearn.tree import DecisionTreeClassifier
max_depth = [10, 50, 100]
min_samples_split = [2,4,6] 

h_param_comb = [{'max_depth':g, 'min_samples_split':c} for g in max_depth for c in min_samples_split]

assert len(h_param_comb) == len(gamma_list)*len(c_list)

dt_accuracy = []


for i in range(0,5):
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=dev_test_frac, shuffle=True
    )
    X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
    )
    for cur_h_params in h_param_comb:

        #PART: Define the model
        # Create a classifier: a support vector classifier
        clf = DecisionTreeClassifier()

        #PART: setting up hyperparameter
        hyper_params = cur_h_params
        clf.set_params(**hyper_params)


        #PART: Train model
        # 2.a train the model 
        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        # print(cur_h_params)
        #PART: get dev set predictions
        predicted_dev = clf.predict(X_test)

        # 2.b compute the accuracy on the validation set
        cur_acc = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_test)
        

        # 3. identify the combination-of-hyper-parameter for which validation set accuracy is the highest. 
        if cur_acc > best_acc:
            best_acc = cur_acc
            best_model = clf
            best_h_params = cur_h_params
            #print("Found new best acc with :"+str(cur_h_params))
            #print("New best val accuracy:" + str(cur_acc)
    dt_accuracy.append(cur_acc)
#print(dt_accuracy)


sreport = pd.DataFrame()
sreport['Accuracy of SVM'] = svm_accuracy
print(sreport)
print("Mean: ",np.mean(svm_accuracy))
print("Std: ",np.std(svm_accuracy))
dreport = pd.DataFrame()
dreport['Accuracy of DTree'] = dt_accuracy
print("\n")
print(dreport)
print("Mean: ",np.mean(dt_accuracy))
print("Std: ",np.std(dt_accuracy))











































