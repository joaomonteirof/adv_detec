from __future__ import print_function
import argparse
import numpy as np
import scipy.linalg as sla
from sklearn import manifold
import matplotlib.pyplot as plt
import pickle
import glob

# Training settings
parser = argparse.ArgumentParser(description='Adversarial/clean Cifar10 samples')
parser.add_argument('--data-path', type=str, default='./', metavar='Path', help='Path for outputing .hdf')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--plot', action='store_true', default=False, help='Plot tsnes')
args = parser.parse_args()

tsne = manifold.TSNE(n_components=2, init='pca')

files_list = glob.glob(args.data_path+'*.p')

if len(files_list)<1:
	raise ValueError('Nothing found at {}.'.format(args.path_to_data))

for i, file_ in enumerate(files_list):

	attack = file_.split('/')[-1].split('_')[-1].split('.')[0]

	pfile = open(file_, 'rb')
	data = pickle.load(pfile)
	pfile.close()

	X_m1 = data[np.where(data[:, -1]==1)[0],:10]
	X_m2 = data[np.where(data[:, -1]==1)[0],10:20]

	m_m1 = X_m1.mean(0)
	C_m1 = np.cov(X_m1, rowvar=False)

	m_m2 = X_m2.mean(0)
	C_m2 = np.cov(X_m2, rowvar=False)

	fid = ((m_m1 - m_m2) ** 2).sum() + np.matrix.trace(C_m1 + C_m2 - 2 * sla.sqrtm(np.matmul(C_m1, C_m2)))

	print(attack, fid)

	if args.plot:
		X_tsne = tsne.fit_transform(np.concatenate([X_m1, X_m2], 0))
		plt.figure(i+1)
		plt.plot(X_tsne[:X_m1.shape[0],0], X_tsne[:X_m1.shape[0],1], 'o', c = 'b', label = 'M1')
		plt.plot(X_tsne[X_m1.shape[0]:,0], X_tsne[X_m1.shape[0]:,1], 'o', c = 'r', label = 'M2')
		plt.title(attack)

X_c_m1 = data[np.where(data[:, -1]==0)[0],:10]
X_c_m2 = data[np.where(data[:, -1]==0)[0],10:20]

m_c_m1 = X_c_m1.mean(0)
C_c_m1 = np.cov(X_c_m1, rowvar=False)

m_c_m2 = X_c_m2.mean(0)
C_c_m2 = np.cov(X_c_m2, rowvar=False)

fid_c = ((m_c_m1 - m_c_m2) ** 2).sum() + np.matrix.trace(C_c_m1 + C_c_m2 - 2 * sla.sqrtm(np.matmul(C_c_m1, C_c_m2)))

print('Clean data', fid_c)

if args.plot:
	X_tsne = tsne.fit_transform(np.concatenate([X_c_m1, X_c_m2], 0))
	plt.figure(i+2)
	plt.plot(X_tsne[:X_c_m1.shape[0],0], X_tsne[:X_c_m1.shape[0],1], 'o', c = 'b', label = 'M1')
	plt.plot(X_tsne[X_c_m1.shape[0]:,0], X_tsne[X_c_m1.shape[0]:,1], 'o', c = 'r', label = 'M2')
	plt.title('Clean data')
	plt.show()
