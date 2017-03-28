import pandas as pd
import  matplotlib as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn import decomposition
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


df = pd.read_table('C:\Users\dcc\Desktop\Tempus/takehome1.txt')     # 530 x 16563
#print(df)

Xdata = df.ix[:,1:].copy()
GTdata = df.ix[:,0].copy()

#Split the Data and Ground Truth into Train Test and Validation sets
# Train = 80% Test = 20%
skf = StratifiedKFold(n_splits = 2)
for Mtrain, test in skf.split(Xdata,GTdata):
        Xtrain = Xdata.ix[Mtrain,:].copy()
        Xtest = Xdata.ix[test,:].copy()
        GTtrain = GTdata[Mtrain]
        GTtest = GTdata[test]


names = ["Logistic Regression","Nearest Neighbors", "Linear SVM", "RBF SVM","Decision Tree", "Random Forest", "NNet"]
classifiers = [ linear_model.LogisticRegression(C= 0.1),
                        KNeighborsClassifier(3),
                        SVC(kernel="linear", C=0.025),
                        SVC(C=0.1, gamma= 0.2),
                        DecisionTreeClassifier(),
                        RandomForestClassifier(max_depth=5),
                        MLPClassifier(alpha=1,hidden_layer_sizes=(20,20))
                        ]
i= 0    #generic iterator

mse_test = np.zeros(len(names))
test_acc = np.zeros(len(names))
cnf_matrix = []
Fmeasure = np.zeros(len(names))
#Initializing Validation and Test metrics
best_val_acc = 0.00
least_val_mse = 1.00
best_test_acc = 0.00
least_test_mse = 1.00

for name, classifier in zip(names, classifiers):

    #Split the Training data into Train and Validation sets
    # Train_actual = 80% of Train data
    # Validation = 20% of Train data.
    # Using Kfold Cross Validation to Split Training data.
    skf1 = StratifiedKFold(n_splits = 2)
    for train, val in skf1.split(Xtrain,GTtrain):
                Xtrain_actual = Xtrain.ix[train,:]
                Xval = Xtrain.ix[val,:]
                GTtrain_actual = GTtrain[train]
                GTval = GTtrain[val]

                pca = decomposition.PCA(n_components=10)
                # fits PCA, transforms data and fits the decision tree classifier
                # on the transformed data
                pipe = Pipeline([('pca', pca),
                                 ('classifier', classifier)])
                pipe.fit(Xtrain_actual,GTtrain_actual)
                predicted = pipe.predict(Xval)

                mse_val = mean_squared_error(GTval, predicted)
                val_acc = accuracy_score(GTval, predicted)
                #print("Mean squared error (Validation) : %.5f" % mse_val )

                if best_val_acc < val_acc:
                    best_val_acc=val_acc

                if mse_val< least_val_mse:
                    least_val_mse = mse_val

    print("Mean squared error (Validation) for "+str(name) + ": " + str(least_val_mse)  )
    print("Validation Accuracy "+str(name) + ": " + str(best_val_acc))
    predicted = pipe.predict(Xtest)

    mse_test[i] = mean_squared_error(GTtest, predicted)  #or use built-in function
    test_acc[i] = accuracy_score(GTtest, predicted)
    print("Mean squared error (Test) for "+str(name)+": "+str(mse_test[i]))
    print("Test Accuracy for " + str(name) + ": " + str(test_acc[i]))

    cnf_matrix.append(confusion_matrix(GTtest,predicted))
    print("Confusion Matrix for "+str(name)+" :")
    print(cnf_matrix[i])

    precision = precision_score(GTtest,predicted)
    recall = recall_score(GTtest,predicted)
    print("Precision:" + str(precision) +" Recall:" +str(recall))
    Fmeasure[i] = f1_score(GTtest, predicted)
    print("F-Measure = "+str(Fmeasure[i]))

    if best_test_acc < test_acc[i]:
        best_test_acc = test_acc[i]


    if mse_test[i] < least_test_mse:
        least_test_mse = mse_test[i]
        best_classifier = name

    i=i+1

print("Best Classifier is : " + str(best_classifier)+"\nMSE = "+str(least_test_mse) + "\nTest Accuracy = "+str(best_test_acc) )