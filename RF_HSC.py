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

from sklearn import cross_validation,datasets,svm
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# Setup parameters
num_labels = 1;

reload(rdHSC)
mag_flag =2
# Load Training Data
TA = rdHSC.sort_data(mag_flag)
TAall = np.array([TA['g_r'],TA['r_z'],TA['g_z'],TA['g_W1'],TA['r_W1'],
                TA['z_W1'],TA['g_W2'],TA['r_W2'],TA['z_W2'],TA['W1_W2'],TA['r'],TA['y']])

TAall = TAall.transpose()
X  = TAall[:,:-1]
y  = TAall[:,-1:]
# clf = MLPClassifier(activation='logistic',solver='adam', alpha=1e-2, hidden_layer_sizes=(100, 2), verbose=True, random_state=1)
# clf.fit(X, y)
print('\n Now test it\n')                                                                                    
TD = rdHSC.sort_test_data(mag_flag)                      
TDall = np.array([TD['g_r'],TD['r_z'],TD['g_z'],TD['g_W1'],TD['r_W1'],TD['z_W1'],TD['g_W2'],TD['r_W2'],TD['z_W2'],TD['W1_W2'],TD['r'], TD['y']])
TestData = TDall.transpose() 
# print(TestData.shape)      
# print("udated\n")
X_test = TestData[:,:-1]  
y_test = TestData[:,-1]
np.random.seed(0)
TP=[]
TN=[]
FP=[]
FN=[]
prec=[]
recall=[]
NF=[]
MDepth=[]
MLeaf=[]
NbForests =[10, 20, 100, 500]
max_depth = [10,15,50,100]
max_ln =[100,1000]

for NbForest in NbForests:
    for md in max_depth:
        for mln in max_ln:
            print(mln,md,NbForest)
            rf = RandomForestClassifier(n_estimators=NbForest, min_samples_split=2, max_depth=md, max_leaf_nodes=mln)       
            rf.fit(X, y.ravel())
            b = rf.predict(X_test)
            yones  = y_test[y_test==1]
            yzeros  = y_test[y_test==0]
            NNones = b[y_test==1]
            NNzeros = b[y_test==0]

            cTP = yones==NNones
            cTN = yzeros=NNzeros
            cFP = sum(NNzeros)
            cFN = len(NNones)-sum(NNones)

            TP.append(sum(cTP))
            TN.append(sum(cTN))
            FP.append(cFP)
            FN.append(cFN)
            prec.append(float(sum(cTP))/float(sum(cTP)+ cFP))
            recall.append(float(sum(cTP))/float(sum(cTP)+ cFN))
            NF.append(NbForest)
            MDepth.append(md)
            MLeaf.append(mln)
            
fileout = 'params_RF_new.txt'
np.savetxt(fileout, np.c_[NF,MDepth,MLeaf,TP,TN,FP,FN,prec,recall],fmt='%1.4f')

#plot ROC curve and compute AUC
from sklearn import metrics
probs = rf.predict_proba(X_test)
fpr, tpr, _ = metrics.roc_curve(y_test, probs[:,1])
import matplotlib.pyplot as plt
plt.plot(fpr,tpr)
plt.fill_between(fpr, 0, tpr,facecolor='powderblue')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC Curve for random forest w/AUC =0.89')
plt.savefig('ROC_RF.pdf')
auc = metrics.auc(fpr,tpr)
'''
# xdat = rf.predict_proba(X_test)
# ydat = clf.predict_proba(X_test)
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
'''quick test plot with subset of data
xrange=xdat[0:10000,1]
yrange=ydat[0:10000,1]
yval=y_test[0:10000]
plt.plot(xrange[yval>0],yrange[yval>0],'ob')
plt.ylabel('NN_results')
plt.xlabel('RF_results')
plt.title('True ELGs')
plt.savefig('ELG.png')'''