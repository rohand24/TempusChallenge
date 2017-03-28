import pandas as pd
import  matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score



df = pd.read_table('C:\Users\dcc\Desktop\DeepLearning\Tempus/tempus/takehome1.txt')     # 530 x 16563
#print(df)

Xdata = df.ix[:,1:].copy()
GTdata = df.ix[:,0].copy()

names = ["Logistic Regression","Nearest Neighbors", "Linear SVM", "Decision Tree", "Random Forest", "NNet"]
classifiers = [
                        linear_model.LogisticRegression(C= 0.1),
                        KNeighborsClassifier(n_neighbors=3,weights='distance', algorithm='auto'),
                        SVC(kernel="linear", C=0.025),
                        DecisionTreeClassifier(max_depth=5),
                        RandomForestClassifier(max_depth=5),
                        MLPClassifier(alpha=0.1,hidden_layer_sizes=(20,20))

                        ]

best_classifier = {}
#initialize metrics

for name, classifier in zip(names, classifiers):

    best_precision = 0.00
    best_recall = 0.00
    best_Fmeasure = 0.00
    best_accuracy = 0.00

    skf = StratifiedKFold(n_splits = 5)
    for Mtrain, test in skf.split(Xdata,GTdata):
        Xtrain = Xdata.ix[Mtrain,:].copy()
        Xtest = Xdata.ix[test,:].copy()
        GTtrain = GTdata[Mtrain]
        GTtest = GTdata[test]



        kf = KFold(n_splits = 5)
        for train, val in kf.split(Xtrain,GTtrain):
                Xtrain_actual = Xtrain.ix[train,:].copy()
                Xval = Xtrain.ix[val,:].copy()
                GTtrain_actual = GTtrain.ix[train]
                GTval = GTtrain[val]

                pca = PCA(n_components=10,whiten=True)
                # fits PCA, transforms data and fits the decision tree classifier
                # on the transformed data
                pipe = Pipeline([('pca', pca),
                                 ('classifier', classifier)])
                pipe.fit(Xtrain, GTtrain)
                predicted = pipe.predict(Xtest)
                precision = precision_score(GTtest, predicted)
                if precision > best_precision:  best_precision=precision
                recall = recall_score(GTtest, predicted)
                if recall > best_recall:  best_recall = recall
                accuracy = accuracy_score(GTtest, predicted)
                if accuracy > best_accuracy:  best_accuracy = accuracy
                Fmeasure = f1_score(GTtest, predicted)
                if Fmeasure > best_Fmeasure:  best_Fmeasure = Fmeasure

    best_classifier[best_Fmeasure] = name

    print ("Classifier: " +str(name))
    print ("Accuracy: " + str(round(best_accuracy,3)))
    print("Precision:" + str(round(best_precision,3)) + "\nRecall: " + str(round(best_recall,3)))
    print("F-Measure = " + str(round(best_Fmeasure,3))+"\n")

best = best_classifier.keys()
Label = best_classifier.values()
r = np.arange(1,len(best)+1,1)
best_Fmeasure = max(best)
print("Best Model is: " + str(best_classifier[best_Fmeasure]))

#sifier.values(),best_classifier.keys(),'ro')
# plt.xlabel('Classifier')
# plt.ylabel('F-measure')
# plt.title('Classifier Comparison')
# plt.legend()
import itertools
marker = itertools.cycle(('ro', 'bs', 'y^', 'bo', 'y*','b^'))
fig, ax = plt.subplots()
for i in range(len(r)):
    ax.plot(r[i], best[i], marker.next(), label=Label[i])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

