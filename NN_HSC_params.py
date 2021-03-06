#NB to use
# % source /project/projectdirs/desi/software/desi_environment.csh
#% pip install --user scikit-learn
from sklearn.neural_network import MLPClassifier
import scipy.misc
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import NN_functions_param as NNp
import readin_HSC as rdHSC
import importlib

# Setup parameters
input_layer_size  = 11;
'''
HLs = [1]
HLn=[10]
alpha = [0.1]
'''
HLs = [4]
HLn=[50]
alpha = [0.001]
svers = ['lbfgs','sgd','adam']

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
TP=[]
TN=[]
FP=[]
FN=[]
prec=[]
recall=[]
av=[]
HL=[]
HLN=[]
solver = svers[2]
print('\n Now test it\n')                                                                                    
TD = rdHSC.sort_test_data(mag_flag)

TDall = np.array([TD['g_r'],TD['r_z'],TD['g_z'],TD['g_W1'],TD['r_W1'],
      TD['z_W1'],TD['g_W2'],TD['r_W2'],TD['z_W2'],TD['W1_W2'],TD['r'], TD['y']])
TestData = TDall.transpose() 
print(TestData.shape)      
X_test = TestData[:,:-1]  
y_test = TestData[:,-1]

for alphaval in alpha:
    for HLsize in HLs:
        for HLnodes in HLn: 
            clf = MLPClassifier(activation='logistic',solver = solver, alpha=alphaval, hidden_layer_sizes=(HLsize, HLnodes), verbose=True, random_state=1)
            clf.fit(X, y.ravel())
            b2 = clf.predict(X_test)
            
            yones  = y_test[y_test==1]
            yzeros  = y_test[y_test==0]
            NNones = b2[y_test==1]
            NNzeros = b2[y_test==0]

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
            av.append(alphaval)
            HL.append(HLsize)
            HLN.append(HLnodes)

fileout = 'params_NN_test'  + '.txt'
np.savetxt(fileout, np.c_[av,HL,HLN,TP,TN,FP,FN,prec,recall],fmt='%1.4f')
from sklearn import metrics
b2_prob = clf.predict_proba(X_test)
fpr, tpr, _ = metrics.roc_curve(y_test, b2_prob[:,1])
auc = metrics.auc(fpr,tpr)
import matplotlib.pyplot as plt
plt.plot(fpr,tpr)
plt.fill_between(fpr, 0, tpr,facecolor='powderblue')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC Curve for NN w/AUC =0.84')
plt.savefig('ROC_NN.pdf')










