import numpy as np
from sklearn import preprocessing, manifold
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import scipy
import matplotlib.pyplot as plt
import pickle

from pylab import rcParams
rcParams['figure.figsize'] = 15, 10

###############################################################################
# Import data set
pfile = open('detec.p', 'rb')
data = pickle.load(pfile)
pfile.close()

X=data[:, :-4]
y=data[:, -4]

#Scaling Data Set into [0,1]

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

###############################################################################
# Instantiate and train clustering estimator

tsne = manifold.TSNE(n_components=2, init='pca')
X_tsne = tsne.fit_transform(X)

clean = []
adv = []

for j in range(y.size):
	if y[j] == 0:
		clean.append(X_tsne[j])
	else:
		adv.append(X_tsne[j])

clean = np.asarray(clean)
adv = np.asarray(adv)

try:
	plt.plot(clean[:,0], clean[:,1],'o' , c = 'b', label = 'Clean')
except IndexError:
	pass
try:
	plt.plot(adv[:,0], adv[:,1],'o' , c = 'r', label = 'Attack')
except IndexError:
	pass

plt.legend()
plt.savefig('tsne_rawdata.png')
plt.clf()
