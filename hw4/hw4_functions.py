#!/usr/bin/env python
""" functions used for AMATH 582 hw4
"""
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import svm

def svd(data, data_axis=1, alpha=0.05):
	[nimgs, nx, ny] = np.shape(data)
	#X = data - np.expand_dims(np.mean(data, axis=data_axis), axis=data_axis)
	X = np.reshape(data, (nimgs, nx*ny), order='F')
	X = X - np.expand_dims(np.mean(X, axis=1), axis=1)

	U, Sdiag, VH = np.linalg.svd(X.T, full_matrices=False) 
	V = VH.T
	Snorm = 100*Sdiag**2/np.sum(Sdiag**2)
	maxind = np.argmin(abs(Snorm-alpha*Snorm[0]))

	Xrec = np.matmul(np.matmul(U[:,:maxind], np.diag(Sdiag[:maxind])), VH[:maxind,:])
	Xrec = np.reshape(Xrec, (nx, ny, nimgs), order='F')	
	return U, Sdiag, V, X, Snorm, maxind, Xrec

def lda(X_train, label_train, X_test, label_test):
	for i in range(len(X_train)):
		X_train[i,:] = np.squeeze(StandardScaler().fit_transform(X_train[i,:].reshape(-1,1)))
	clf = LDA(solver='svd')
	clf.fit(X_train, label_train)

	X_transform = clf.transform(X_train)

	y_predict = np.copy(label_test)
	for i in range(len(label_test)):
		y_predict[i] = clf.predict(X_test[i,:].reshape(1,-1))[0]
	matches = np.where(y_predict==label_test)[0]
	return len(matches)/len(y_predict)

def dct(X_train, label_train, X_test, label_test):	
	dt = DecisionTreeClassifier()
	dt.fit(X_train, label_train)
	y_predict = dt.predict(X_test)
	matches = np.where(y_predict==label_test)[0]	
	cm = confusion_matrix(label_test, y_predict, labels=np.unique(label_test))
	cm_norm = cm/np.expand_dims(np.sum(cm, axis=0), axis=0)
	cm_norm2 = cm/np.expand_dims(np.sum(cm, axis=1), axis=1)	
	return len(matches)/len(y_predict)

def svc(X_train, label_train, X_test, label_test):	
	clf = svm.SVC()
	clf.fit(X_train, label_train)
	y_predict = np.copy(label_test)
	for i in range(len(label_test)):
		y_predict[i] = clf.predict(X_test[i,:].reshape(1,-1))[0]
	matches = np.where(y_predict==label_test)[0]
	return len(matches)/len(y_predict)	

def compare_classify(X_train, label_train, X_test, label_test, digits):
	ind = []
	for digit in digits:
		dum = np.where(label_train==digit)[0]
		for i in range(len(dum)):
			ind.append(dum[i])
	ind_test = []
	for digit in digits:
		dum = np.where(label_test==digit)[0]
		for i in range(len(dum)):
			ind_test.append(dum[i])	
	lda_success = lda(X_train[ind,:], label_train[ind], X_test[ind_test,:], label_test[ind_test])
	dct_success = dct(X_train[ind,:], label_train[ind], X_test[ind_test,:], label_test[ind_test])
	svm_success = svc(X_train[ind,:], label_train[ind], X_test[ind_test,:], label_test[ind_test])
	return lda_success, dct_success, svm_success


