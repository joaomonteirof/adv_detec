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

def calculate_accuracy(model, x_train, y_train, x_test, y_test):

	model.fit(x_train, y_train)
	y_pred = model.predict(x_test)
	acc = metrics.accuracy_score(y_test, y_pred)

	return acc

attacks = ['fgsm', 'igsm', 'jsma', 'deepfool', 'gaussianblur', 'gaussiannoise', 'saltandpepper']

results_rfo = np.zeros([len(attacks), len(attacks)])
results_knn = np.zeros([len(attacks), len(attacks)])
results_svm = np.zeros([len(attacks), len(attacks)])

min_max_scaler = preprocessing.MinMaxScaler()

for i in (range(len(attacks))):
	for j in (range(len(attacks))):

		if i != j:

			print(attacks[i], attacks[j])

			###############################################################################
			# Import data set

			traindata_file = open('detec_mnist_'+attacks[i]+'.p', 'rb')
			train_data = pickle.load(traindata_file)
			traindata_file.close()

			X = train_data[:, :-1]
			y = train_data[:, -1]

			#Scaling Train Data into [0,1]

			X = min_max_scaler.fit_transform(X)

			X_train, Y_train = X, y

			testdata_file = open('detec_mnist_'+attacks[j]+'.p', 'rb')
			test_data = pickle.load(testdata_file)
			testdata_file.close()

			X = test_data[:, :-1]
			y = test_data[:, -1]

			#Scaling Train Data into [0,1]

			X = min_max_scaler.fit_transform(X)

			X_test, Y_test = X, y

			###############################################################################
			# Instantiate models

			forest = RandomForestClassifier(n_estimators=10)
			neigh = KNeighborsClassifier(n_neighbors=3)
			svc_rbf3 = SVC(kernel='rbf', C=100.0, gamma=0.1)

			###############################################################################
			# Printing/Plotting Results

			results_rfo[i,j] = calculate_accuracy(forest, X_train, Y_train, X_test, Y_test)
			results_knn[i,j] = calculate_accuracy(neigh, X_train, Y_train, X_test, Y_test)
			results_svm[i,j] = calculate_accuracy(svc_rbf3, X_train, Y_train, X_test, Y_test)

			print(results_rfo[i,j], results_knn[i,j], results_svm[i,j])

print(results_rfo)
print(results_knn)
print(results_svm)

np.savetxt('mnist_rfo.csv', results_rfo, delimiter=',')
np.savetxt('mnist_knn.csv', results_knn, delimiter=',')
np.savetxt('mnist_svm.csv', results_svm, delimiter=',')
