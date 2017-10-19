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
hidden_layer_size = 10;
it_no =50
num_labels = 1;

importlib.reload(rdHSC)
mag_flag =2
# Load Training Data
TA = rdHSC.sort_data(mag_flag)
if (mag_flag ==1):
    TAall = np.array([TA['g_r'],TA['r_z'],TA['g_z'],TA['g_W1'],TA['r_W1'],
                TA['z_W1'],TA['g_W2'],TA['r_W2'],TA['z_W2'],TA['W1_W2'],TA['g'],TA['y']])
if (mag_flag ==2):
    TAall = np.array([TA['g_r'],TA['r_z'],TA['g_z'],TA['g_W1'],TA['r_W1'],
                TA['z_W1'],TA['g_W2'],TA['r_W2'],TA['z_W2'],TA['W1_W2'],TA['r'],TA['y']])
TAall = TAall.transpose()
X  = TAall[:,:-1]
y  = TAall[:,-1:]

clf = MLPClassifier(activation='logistic',solver='adam', alpha=1e-2, hidden_layer_sizes=(50, 4), verbose=True, random_state=1)
clf.fit(X, y)

print('\n Now test it\n')                                                                                    
TD = rdHSC.sort_test_data(mag_flag)

TDall = np.array([TD['g_r'],TD['r_z'],TD['g_z'],TD['g_W1'],TD['r_W1'],
                  TD['z_W1'],TD['g_W2'],TD['r_W2'],TD['z_W2'],TD['W1_W2'],TD['r'], TD['y']])
TestData = TDall.transpose() 
print(TestData.shape)      
# print("udated\n")
X_test = TestData[:,:-1]  
y_test = TestData[:,-1]
score=[] 
b = clf.predict(X_test)
# b = np.reshape(b, [len(b),1])
c = y_test==b
score.append(float(sum(c))/len(b))
print(score)

yones  = y_test[y_test==1]
NNones = b[y_test==1]

cNNones = yones==NNones
scoreTrueNN = (float(sum(cNNones))/len(NNones))

print(scoreTrueNN)

yzs  = y_test[y_test==0]
NNwrong = b[y_test==0]

cNNzs = yzs!=NNwrong
scoreFNN = (float(sum(cNNzs))/len(NNwrong))
print(scoreFNN)
##with r mag
# 200:2, 100:4, 100:3, 100:2-lbfgs
# [0.66496, 0.65218, 0.66362, 0.66752]
#100;2 -sgd 0.65028
#100:2 -adam 0.65046

#with g mag
#100:2 -adam [0.67248], -lbfgs 0.47964, -sgd 0.6784
#alpha = 1e-5 0.6753 (sgd,100,2)
#alpha = 1e-4 0.6753 (sgd,100,2)
#alpha = 1e-3 0.6753 (sgd,100,2)
#alpha = 1e-2 0.67554(sgd,100,2)
#alpha = 1e-1 0.67506(sgd,100,2)

#alpha = 1e-2 0.68608 (sgd, 100, 3)
#alpha =1e-2 0.68472 (sgd, 200, 2)
#alpha =1e-2 0.88768 (sgd, 200,3)
#alpha =1e-2 0.6811 (sgd, 200,4)
