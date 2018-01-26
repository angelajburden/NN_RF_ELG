from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import scipy.misc
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import metrics
# import NN_functions_param as NNp
import readin_HSC as rdHSC
import importlib

# Setup parameters
n_components = 6;

reload(rdHSC)
mag_flag =2
# Load Training Data
TA = rdHSC.sort_data(mag_flag)
TAall = np.array([TA['g_r'],TA['r_z'],TA['g_z'],TA['g_W1'],TA['r_W1'],
                TA['z_W1'],TA['g_W2'],TA['r_W2'],TA['z_W2'],TA['W1_W2'],TA['r'],TA['y']])
TAall = TAall.transpose()
X  = TAall[:,:-1]
y  = TAall[:,-1:]

print('\n Now test it\n')                                                                                    
TD = rdHSC.sort_test_data(mag_flag)

TDall = np.array([TD['g_r'],TD['r_z'],TD['g_z'],TD['g_W1'],TD['r_W1'],
      TD['z_W1'],TD['g_W2'],TD['r_W2'],TD['z_W2'],TD['W1_W2'],TD['r'], TD['y']])
TestData = TDall.transpose() 
print(TestData.shape)      
X_test = TestData[:,:-1]  
y_test = TestData[:,-1]

#normalise data
scaled_training_X = preprocessing.scale(X)
scaled_test_X = preprocessing.scale(X_test)
pca = PCA(n_components = n_components)
pca.fit(scaled_training_X)
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
# plt.plot(var1)
# plt.plot(var1, '.')
# plt.xlabel('PCs')
# plt.ylabel('Cumalitive Variance')
# plt.title('Principle Component Analysis in 6 directions')
# plt.savefig('PCA6.pdf')
n_components = 4
pca = PCA(n_components = n_components)
pca.fit(scaled_training_X)
pca_Xtrain =pca.fit_transform(scaled_training_X)
pca_Xtest = pca.fit_transform(scaled_test_X)

# see if NN does better
alphaval =0.001
svers = 'adam'
HLsize = 4
HLnodes= 50 
clf = MLPClassifier(activation='logistic',solver = svers, alpha=alphaval, hidden_layer_sizes=(HLsize, HLnodes), verbose=True, random_state=1)
clf.fit(pca_Xtrain, y.ravel())
b2 = clf.predict(pca_Xtest)

b2_prob = clf.predict_proba(pca_Xtest)
fpr, tpr, _ = metrics.roc_curve(y_test, b2_prob[:,1])
auc = metrics.auc(fpr,tpr)

plt.plot(fpr,tpr)
plt.fill_between(fpr, 0, tpr,facecolor='powderblue')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC Curve for NN w/AUC =%f' % auc)
plt.savefig('ROC_NN_postPCA.pdf')