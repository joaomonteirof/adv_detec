from collections import OrderedDict
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import scipy
import matplotlib.pyplot as plt
import pickle
import os
import csv

from pylab import rcParams
rcParams['figure.figsize'] = 15, 10

###############################################################################
# Import data set
pfile = open('detec_oltl.p', 'rb')
data = pickle.load(pfile)
pfile.close()

X=data[:, :-1]
y=data[:, -1]

#Scaling Data Set into [0,1]

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

###############################################################################
# Instantiate and train clustering estimator

est = KMeans(n_clusters=2)
est.fit(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

symb = ['^', '*']

for i in range(est.n_clusters):
	X_split = X_pca[np.where(est.labels_ == i)]
	y_split = y[np.where(est.labels_ == i)]

	clean = []
	adv = []

	for j in range(y_split.size):	
		if y_split[j] == 0:
			clean.append(X_split[j])
		else:
			adv.append(X_split[j])

	clean = np.asarray(clean)
	adv = np.asarray(adv)

	try:
		plt.plot(clean[:,0], clean[:,1], symb[i], c = 'b', label = 'Clean, cluster'+' '+str(i))
	except IndexError:
		pass
	try:
		plt.plot(adv[:,0], adv[:,1], symb[i], c = 'r', label = 'Attack, cluster'+' '+str(i))
	except IndexError:
		pass

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.savefig('kmeans_oltl.png')
plt.clf()
