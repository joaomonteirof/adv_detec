import numpy as np
from sklearn import preprocessing, tree, metrics, linear_model
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pickle

from pylab import rcParams
rcParams['figure.figsize'] = 15, 10

def calculate_metrics(model, x_train, y_train, x_test, y_test, k=5):

	tp_class = 0

	idx = np.split(np.arange(x_test.shape[0]), k)

	model.fit(x_train, y_train)

	acc = []
	precision = []
	recall = []
	f1 = []
	auc = []

	for idx_ in idx:

		x_test_, y_test_ = x_test[idx_, :], y_test[idx_]

		y_pred, y_score = model.predict(x_test_), model.predict_proba(x_test_)[:,1]

		acc.append( metrics.accuracy_score(y_test_, y_pred) )
	
		fpr, tpr, thresholds = metrics.roc_curve(y_test_, y_score)
		auc.append( metrics.auc(fpr, tpr) )

		precision.append( metrics.precision_score(y_test_, y_pred, pos_label = tp_class, average = 'binary') )

		recall.append( metrics.recall_score(y_test_, y_pred, pos_label = tp_class, average = 'binary') )

		f1.append( metrics.f1_score(y_test_, y_pred, pos_label = tp_class, average = 'binary') )

	return [np.mean(acc), np.mean(precision), np.mean(recall), np.mean(f1), np.mean(auc), np.std(acc), np.std(precision), np.std(recall), np.std(f1), np.std(auc)]

'''

def calculate_metrics(model, x_train, y_train, x_test, y_test):

	tp_class = 0

	model.fit(x_train, y_train)

	y_pred, y_score = model.predict(x_test), model.predict_proba(x_test)[:,1]

	acc = metrics.accuracy_score(y_test, y_pred)
	
	fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
	auc = metrics.auc(fpr, tpr)

	precision = metrics.precision_score(y_test, y_pred, pos_label = tp_class, average = 'binary')

	recall = metrics.recall_score(y_test, y_pred, pos_label = tp_class, average = 'binary')

	f1 = metrics.f1_score(y_test, y_pred, pos_label = tp_class, average = 'binary')

	return [acc, precision, recall, f1, auc]
'''

###############################################################################
# Import data set

pfile = open('detec_saltandpepper.p', 'rb')
data = pickle.load(pfile)
pfile.close()

X=data[:, :-1]
y=data[:, -1]

#Scaling Data Set into [0,1]

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

X_train, Y_train = X, y

pfile_ = open('detec_deepfool.p', 'rb')
data_ = pickle.load(pfile_)
pfile_.close()

X_=data_[:, :-1]
y_=data_[:, -1]

#Scaling Data Set into [0,1]

X_ = min_max_scaler.fit_transform(X_)

X_test, Y_test = X_, y_

#Y_test=np.round(np.random.rand(Y_test.shape[0]))

###############################################################################
# Instantiate models

forest = RandomForestClassifier(n_estimators=10)
neigh = KNeighborsClassifier(n_neighbors=3)
svc_rbf3 = SVC(kernel='rbf', C=100.0, gamma=0.1, probability=True)

###############################################################################
# Printing/Plotting Results

scores_rf = calculate_metrics(forest, X_train, Y_train, X_test, Y_test)
scores_neigh = calculate_metrics(neigh, X_train, Y_train, X_test, Y_test)
scores_rbf3 = calculate_metrics(svc_rbf3, X_train, Y_train, X_test, Y_test)

print('Random Forest scores - Valid:')
print(scores_rf)

print('KNN scores - Valid:')
print(scores_neigh)

print('SVMs scores - Valid:')
print(scores_rbf3)

modelsList = ['RF', 'KNN', 'SRBF3']
accuraciesList = [scores_rf, scores_neigh, scores_rbf3]

N = 3
ind = np.arange(N)  # the x locations for the groups
width = 0.15      # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

acc_values = [scores_rf[0], scores_neigh[0], scores_rbf3[0]]
precision_values = [scores_rf[1], scores_neigh[1], scores_rbf3[1]]
recall_values = [scores_rf[2], scores_neigh[2], scores_rbf3[2]]
f1_values = [scores_rf[3], scores_neigh[3], scores_rbf3[3]]
auc_values = [scores_rf[4], scores_neigh[4], scores_rbf3[4]]

rects1 = ax.bar(ind, acc_values, width, color='r')
rects2 = ax.bar(ind+width, precision_values, width, color='g')
rects3 = ax.bar(ind+width*2, recall_values, width, color='b')
rects4 = ax.bar(ind+width*3, f1_values, width, color='y')
rects5 = ax.bar(ind+width*4, auc_values, width, color='k')

ax.set_ylabel('Scores')
ax.set_xticks(ind+width)
ax.set_xticklabels(modelsList)
ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]), ('Accuracy', 'Precision', 'Recall', 'F1', 'AUC') )

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%.2f'%float(h),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)

plt.savefig('./detectors_saltandpepper.png')
plt.show()
