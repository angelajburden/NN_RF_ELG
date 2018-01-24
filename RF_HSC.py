#NB to use
# % source /project/projectdirs/desi/software/desi_environment.csh
#% pip install --user scikit-learn
from sklearn.neural_network import MLPClassifier
import scipy.misc
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import readin_HSC as rdHSC
import importlib
from sklearn import cross_validation,datasets,svm
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# Setup parameters
input_layer_size  = 11;
hidden_layer_size = 10;
it_no =50
num_labels = 1;
NbForest =500
importlib.reload(rdHSC)
mag_flag =2
# Load Training Data
TA = rdHSC.sort_data(mag_flag)
TAall = np.array([TA['g_r'],TA['r_z'],TA['g_z'],TA['g_W1'],TA['r_W1'],
                TA['z_W1'],TA['g_W2'],TA['r_W2'],TA['z_W2'],TA['W1_W2'],TA['r'],TA['y']])

TAall = TAall.transpose()
X  = TAall[:,:-1]
y  = TAall[:,-1:]

rf = RandomForestClassifier(n_estimators=NbForest, min_samples_split=2, max_depth=15, max_leaf_nodes=2000)
np.random.seed(0)
rf.fit(X, y)
clf = MLPClassifier(activation='logistic',solver='adam', alpha=1e-2, hidden_layer_sizes=(100, 2), verbose=True, random_state=1)
clf.fit(X, y)

print('\n Now test it\n')                                                                                    
TD = rdHSC.sort_test_data(mag_flag)
                      
TDall = np.array([TD['g_r'],TD['r_z'],TD['g_z'],TD['g_W1'],TD['r_W1'],TD['z_W1'],TD['g_W2'],TD['r_W2'],TD['z_W2'],TD['W1_W2'],TD['r'], TD['y']])

TestData = TDall.transpose() 
print(TestData.shape)      
# print("udated\n")
X_test = TestData[:,:-1]  
y_test = TestData[:,-1]
scoreRF=[]
scoreNN=[] 
bRF = rf.predict(X_test)
bNN = clf.predict(X_test)
# b = np.reshape(b, [len(b),1])
cRF = y_test==bRF
cNN = y_test==bNN
scoreRF.append(float(sum(cRF))/len(bRF))
scoreNN.append(float(sum(cNN))/len(bNN))

print("score RF =%f\n"% scoreRF)
print("score NN=%f\n"%scoreNN)
xdat = rf.predict_proba(X_test)
ydat = clf.predict_proba(X_test)

'''quick test plot with subset of data
xrange=xdat[0:10000,1]
yrange=ydat[0:10000,1]
yval=y_test[0:10000]
plt.plot(xrange[yval>0],yrange[yval>0],'ob')
plt.ylabel('NN_results')
plt.xlabel('RF_results')
plt.title('True ELGs')
plt.savefig('ELG.png')'''

yones  = y_test[y_test==1]
RFones = bRF[y_test==1]
NNones = bNN[y_test==1]

cRFones = yones==RFones
cNNones = yones==NNones
scoreTrueNN = (float(sum(cNNones))/len(NNones))
scoreTrueRF = (float(sum(cRFones))/len(RFones))

print(scoreTrueRF, scoreTrueNN)

yzs  = y_test[y_test==0]
RFwrong = bRF[y_test==0]
NNwrong = bNN[y_test==0]

cRFzs = yzs!=RFwrong
cNNzs = yzs!=NNwrong
scoreFNN = (float(sum(cNNzs))/len(NNwrong))
scoreFRF = (float(sum(cRFzs))/len(RFwrong))

print(scoreFRF, scoreFNN)

avdat = xdat[:,1]+ydat[:,1]
avdat = avdat/2
avdat[avdat>0.5]=1
avdat[avdat<0.5]=0
cav= y_test==avdat
scav=float(sum(cav))/len(avdat)
print(scav)
avone=avdat[y_test==1]
Fones = yones==avone
print(float(sum(Fones))/len(yones))

#plot ROC curve and compute AUC
from sklearn import metrics
import matplotlib.pyplot as plt

probs = rf.predict_proba(X_test)
fpr, tpr, _ = metrics.roc_curve(y_test, probs[:,1])
auc = metrics.auc(fpr,tpr)
plt.plot(fpr,tpr)
plt.fill_between(fpr, 0, tpr,facecolor='powderblue')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC Curve for random forest w/AUC =%f' %auc)
plt.savefig('ROC_RF.pdf')
plt.close()

b2_prob = clf.predict_proba(X_test)
fpr, tpr, _ = metrics.roc_curve(y_test, b2_prob[:,1])
auc = metrics.auc(fpr,tpr)
plt.plot(fpr,tpr)
plt.fill_between(fpr, 0, tpr,facecolor='powderblue')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC Curve for NN w/AUC =%f' %auc)
plt.savefig('ROC_NN.pdf')

'''
#histograms of results for both random forest and NN methods
plt.figure()
mask_TP = np.zeros(len(y_test), bool) | ((y_test>0) & (bRF >0)) #valnn=0 TP
mask_FN = np.zeros(len(y_test), bool) | ((y_test==0) & (bRF>0))   #valnn=1 FN
mask_TN = np.zeros(len(y_test), bool) | ((y_test==0) & (bRF == 0))   #valnn=2 TN
mask_FP = np.zeros(len(y_test), bool) | ((y_test>0) & (bRF == 0))  #valnn=3 FP

plt.hist(xdat[mask_TP,1], \
         bins=np.arange(0.-0.02,1. + 0.02, 0.02), \
         ec='blue', fc='none', lw=1.5, histtype='step', label='TP' )
plt.hist(xdat[mask_TN,1], \
         bins=np.arange(0.-0.02,1. + 0.02, 0.02), \
         ec='red', fc='none', lw=1.5, histtype='step', label='TN' )
plt.hist(xdat[mask_FP,1], \
         bins=np.arange(0.-0.02,1. + 0.02, 0.02), \
         ec='black', fc='none', lw=1.5, histtype='step', label='FP' )
plt.hist(xdat[mask_FN,1], \
         bins=np.arange(0.-0.02,1. + 0.02, 0.02), \
         ec='green', fc='none', lw=1.5, histtype='step', label='FN' )
plt.legend(loc='upper center') 
plt.ylabel("count")
plt.title('RF scores') 
plt.savefig('histRF.png')
plt.figure()
mask_TP = np.zeros(len(y_test), bool) | ((y_test>0) & (bNN >0)) #valnn=0 TP                                                                             
mask_FN = np.zeros(len(y_test), bool) | ((y_test==0) & (bNN>0))   #valnn=1 FN                                                                             
mask_TN = np.zeros(len(y_test), bool) | ((y_test==0) & (bNN == 0))   #valnn=2 TN                                                                         
mask_FP = np.zeros(len(y_test), bool) | ((y_test>0) & (bNN == 0))  #valnn=3 FP                                                                           
plt.hist(ydat[mask_TP,1], \
         bins=np.arange(0.-0.02,1. + 0.02, 0.02), \
         ec='blue', fc='none', lw=1.5, histtype='step', label='TP' )
plt.hist(ydat[mask_TN,1], \
         bins=np.arange(0.-0.02,1. + 0.02, 0.02), \
         ec='red', fc='none', lw=1.5, histtype='step', label='TN' )
plt.hist(ydat[mask_FP,1], \
         bins=np.arange(0.-0.02,1. + 0.02, 0.02), \
         ec='black', fc='none', lw=1.5, histtype='step', label='FN' )
plt.hist(ydat[mask_FN,1], \
         bins=np.arange(0.-0.02,1. + 0.02, 0.02), \
         ec='green', fc='none', lw=1.5, histtype='step', label='FP' )
plt.legend(loc='upper center') 
plt.ylabel("count")
plt.title('NN scores') 
plt.savefig('histNN.png')

#tests to check which data points were classified the same by both methods 
#shows contour plot of RF prediction[ith data] vs NN prediction[ith data]
#for subset of data points

import seaborn as sns
sns.set(color_codes=True)
f, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
ax[0].set_ylabel('NN')
ax[0].set_xlabel('RF')
ax[1].set_xlabel('RF')
sns.kdeplot(xdat[y_test==1,1], ydat[y_test==1,1], cmap="Reds",n_levels=100, ax= ax[0],cbar=True,label='True ELG', legend=True)
sns.kdeplot(xdat[y_test==0,1], ydat[y_test==0,1], cmap="Blues",n_levels=100, ax=ax[1],cbar=True, label='Not ELG', legend=True)
sns.set_style("ticks")
f.tight_layout()
f.savefig("contour.png")

'''
