import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from scipy import linalg
import os
import hw4_functions as f 
plt.ion()
plt.style.use('ggplot')

(img_train, label_train), (img_test, label_test) = mnist.load_data()
[nimgs_train, nx, ny] = np.shape(img_train)
[nimgs_test, nx, ny] = np.shape(img_test)

### svd of training and test data
U_train, Sdiag_train, V_train, X_train, Snorm_train, maxind_train, Xrec_train = f.svd(img_train, data_axis=1, alpha=0.1)
U_test, Sdiag_test, V_test, X_test, Snorm_test, maxind_test, Xrec_test = f.svd(img_test, data_axis=1, alpha=0.1)

#Xrec_train = np.reshape(Xrec_train, (len(X_train[0,:]), len(X_train))).T

#X_train = Xrec_train
### two digit compare
labels = np.unique(label_train)
success_rates_lda = np.ones((len(labels),len(labels)))
success_rates_dct = np.ones((len(labels),len(labels)))
success_rates_svm = np.ones((len(labels),len(labels)))
for i in range(len(np.unique(label_train))):
	for j in range(len(np.unique(label_train))):
		if i!=j:
			digits = [i, j]
			success_rates_lda[i,j], success_rates_dct[i,j], success_rates_svm[i,j] = f.compare_classify(X_train, label_train, X_test, label_test, digits)
	print('done with %d' % i)


### three digit compare
digits_hard = [3, 5, 8]
success_rates_lda_3hard, success_rates_dct_3hard, success_rates_svm_3hard = f.compare_classify(X_train, label_train, X_test, label_test, digits_hard)
print('done with 1')

digits_easy = [0, 3, 6]
success_rates_lda_3easy, success_rates_dct_3easy, success_rates_svm_3easy = f.compare_classify(X_train, label_train, X_test, label_test, digits_easy)
print('done with 2')
### all digit compare
success_rates_lda_all, success_rates_dct_all, success_rates_svm_all = f.compare_classify(X_train, label_train, X_test, label_test, np.unique(label_train))

print('done with 3')

digits_hard = [3, 5, 8]
train_success_rates_lda_3hard, train_success_rates_dct_3hard, train_success_rates_svm_3hard = f.compare_classify(X_train, label_train, X_train, label_train, digits_hard)
print('done with 4')
digits_easy = [0, 3, 6]
train_success_rates_lda_3easy, train_success_rates_dct_3easy, train_success_rates_svm_3easy = f.compare_classify(X_train, label_train, X_train, label_train, digits_easy)
### all digit compare
print('done with 5')
train_success_rates_lda_all, train_success_rates_dct_all, train_success_rates_svm_all = f.compare_classify(X_train, label_train, X_train, label_train, np.unique(label_train))
print('done with 6')



if 1: ## plot svd spectrum
	plt.style.use('ggplot')
	fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(12,5))
	ax[0].plot(np.arange(1,len(Snorm_train)+1,1), Snorm_train, linewidth=3)
	ax[0].plot(maxind_train, Snorm_train[maxind_train], 'o', markersize=10)
	ax[0].set_ylabel('Percentage of Variance Explained')
	ax[0].set_xlabel('Modes')
	ax[0].set_title('Training Data')
	ax[0].set_xlim(-5,100)

	ax[1].plot(np.arange(1,len(Snorm_test)+1,1), Snorm_test, linewidth=3)
	ax[1].plot(maxind_test, Snorm_test[maxind_test], 'o', markersize=10)
	ax[1].set_xlabel('Modes')
	ax[1].set_title('Test Data')
	ax[1].set_xlim(-5,100)
	fig.savefig('svg_spectrum.png')

if 1: ## plot reconstructed images
	img_num = 15
	fig, ax = plt.subplots(ncols=2, figsize=(9,4.5))
	ax[0].pcolormesh(img_train[img_num,::-1,:], cmap='Greys')
	p = ax[1].pcolormesh(Xrec_train[::-1,:,img_num], cmap='Greys')
	#p.set_clim(0,np.max(Xrec_train[img_num,:,:]))
	ax[0].get_xaxis().set_visible(False)	
	ax[0].get_yaxis().set_visible(False)
	ax[1].get_xaxis().set_visible(False)	
	ax[1].get_yaxis().set_visible(False)
	ax[0].spines['bottom'].set_color('0.5')
	ax[0].spines['top'].set_color('0.5')
	ax[0].spines['right'].set_color('0.5')
	ax[0].spines['left'].set_color('0.5')
	ax[1].spines['bottom'].set_color('0.5')
	ax[1].spines['top'].set_color('0.5')
	ax[1].spines['right'].set_color('0.5')
	ax[1].spines['left'].set_color('0.5')
	ax[0].set_title('Original')
	ax[1].set_title('Reconstructed')
	fig.savefig('img_recon.png')

if 0: ## plot v projection scatter
	plt.style.use('default')
	label_train_str = np.unique([str(label) for label in label_train])
	colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:cyan', 'tab:grey', 'tab:pink', 'tab:olive', 'tab:brown']
	label_colors = [colors[label] for label in label_train]
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.scatter(V_train[:,1], V_train[:,2], V_train[:,4], c=label_colors)
	ax.legend(loc='best')
	fig.savefig('v_modes.png')

if 1: 
	plt.style.use('ggplot')
	fig, ax = plt.subplots(ncols=3, figsize=(15,4))
	p0 = ax[0].imshow(success_rates_lda, cmap='viridis')
	fig.colorbar(p0, ax=ax[0], label='Success Rate')
	p0.set_clim(0.5,1)
	ax[0].set_xticks(range(len(labels)))
	ax[0].set_yticks(range(len(labels)))
	ax[0].set_xticklabels(labels, fontsize=14)
	ax[0].set_yticklabels(labels, fontsize=14)
	ax[0].set_title('LDA')
	ax[0].set_xlabel('Digit')
	ax[0].set_ylabel('Digit')

	p1 = ax[1].imshow(success_rates_dct, cmap='viridis')
	fig.colorbar(p1, ax=ax[1], label='Success Rate')
	p1.set_clim(0.5,1)
	ax[1].set_xticks(range(len(labels)))
	ax[1].set_yticks(range(len(labels)))
	ax[1].set_xticklabels(labels, fontsize=14)
	ax[1].set_yticklabels(labels, fontsize=14)
	ax[1].set_title('DTC')
	ax[1].set_xlabel('Digit')
	ax[1].set_ylabel('Digit')

	p2 = ax[2].imshow(success_rates_svm, cmap='viridis')
	fig.colorbar(p2, ax=ax[2], label='Success Rate')
	p2.set_clim(0.5,1)
	ax[2].set_xticks(range(len(labels)))
	ax[2].set_yticks(range(len(labels)))
	ax[2].set_xticklabels(labels, fontsize=14)
	ax[2].set_yticklabels(labels, fontsize=14)
	ax[2].set_title('SVM')	
	ax[2].set_xlabel('Digit')
	ax[2].set_ylabel('Digit')
	fig.tight_layout()
	fig.savefig('2digit_compare.png')

if 1:
	plt.style.use('ggplot')
	fig, ax = plt.subplots(figsize=(7,3))
	ax.plot(np.arange(3), np.array([success_rates_lda_3hard, success_rates_dct_3hard, success_rates_svm_3hard]), 'o-', color='tab:blue', label='3, 5, 8 digits')
	ax.plot(np.arange(3), np.array([success_rates_lda_3easy, success_rates_dct_3easy, success_rates_svm_3easy]), 'o-', color='tab:green', label='0, 3, 6 digits')
	ax.plot(np.array([success_rates_lda_all, success_rates_dct_all, success_rates_svm_all]), 'o-', color='tab:grey', label='All digits')
	ax.plot(np.arange(3), np.array([train_success_rates_lda_3hard, train_success_rates_dct_3hard, train_success_rates_svm_3hard]), 'o--', color='tab:blue')
	ax.plot(np.arange(3), np.array([train_success_rates_lda_3easy, train_success_rates_dct_3easy, train_success_rates_svm_3easy]), 'o--', color='tab:green')
	ax.plot(np.array([train_success_rates_lda_all, train_success_rates_dct_all, train_success_rates_svm_all]), 'o--', color='tab:grey')
	ax.legend(loc='best')
	ax.set_ylabel('Success Rate')
	ax.set_xticks(np.arange(3))	
	ax.set_xticklabels(['LDA', 'DTC', 'SVM'], fontsize=14)
	fig.savefig('3digit_all_compare.png')



