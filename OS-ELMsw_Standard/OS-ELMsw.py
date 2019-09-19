"""
Created on Tue May 23 13:35:52 2017

@author: Andres
"""
#encoding utf-8
import numpy as np
from time import time
from sklearn.metrics import mean_squared_error
from ActivationFunction import SigActFun



def OSELM(N0, nHiddenNeurons, Block, windowSize, dataSet):
    ##### Load dataset
    try:
    	data = np.genfromtxt(dataSet, dtype = float)
    except Exception:
    	print('An error ecurred while loading data')
    
    
    T = np.array(data[:,0])
    P = np.array(data[:,1:data.shape[1]])
    
    nInputNeurons = data.shape[1]-1
    nTrainingData = data.shape[0]                          
    
    ##### Step 1 Initialization Phase
    P0 = P[0:N0,:]
    T0 = T[0:N0]
    
    PT = np.array(P[N0-windowSize:nTrainingData,:])
    TT = np.array(T[N0-windowSize:nTrainingData])
    Real = np.array(T[N0-1:nTrainingData-1])
    Y = np.zeros_like(Real)
    Error = np.zeros_like(Real)
    
    start_time = time()
    errorSum = 0
    # Input weights randomly chosen from the range [-1,1]
    IW = np.random.uniform(-1, 1, (nHiddenNeurons,nInputNeurons))
    
    # Biases randomly chosen from the range [-1,1]
    Bias = np.random.uniform(-1, 1, (1,nHiddenNeurons))
    H0 = SigActFun(Bias, P0, IW)
    
    M = np.linalg.pinv(np.dot(np.transpose(H0),H0))
    beta = np.dot(np.linalg.pinv(H0),T0)
    
    for n in range(windowSize, PT.shape[0]):  
        Ptemp1 = np.array(PT[(n-windowSize):n])
        Ttemp1 = np.array(TT[(n-windowSize):n])
    
        H = SigActFun(Bias, Ptemp1, IW)
       
        g = np.dot(H,beta)
        Y[n-windowSize] = g[-1]
        
        K = np.dot(M,np.transpose(H))
        Q = np.linalg.inv(np.eye(windowSize) + np.dot(H, K))
        R = np.dot(K, Q)
        S = np.dot(H, M)
        M = M - np.dot(R, S)
        
        beta = beta + np.dot(np.dot(M,np.transpose(H)),(Ttemp1-np.dot(H,beta)))
        
        errorSum += np.power((Ttemp1[-1]-Y[n-windowSize]), 2)
        Error[n-windowSize] = errorSum/((n-windowSize)+1)
    
    ##### Time in seconds
    end_time = time()
    totalTime = end_time - start_time
    hours, rest = divmod(totalTime,3600)
    minutes, seconds = divmod(rest, 60)
    compTime = np.round(seconds,5)
    
    ##### Acurracy MSE
    mse = mean_squared_error(Real, Y)
    accuracy = np.round(mse,5)
    
    last_Real = Real[-200:]
    last_Y = Y[-200:]
    last_accuracy = mean_squared_error(last_Real, last_Y)
    
    return accuracy, compTime, Error, last_accuracy


##### Macro definition
N0 = 120
nHiddenNeurons = 15
windowSize = 3
Block = 1
dataSet = 'Sao_Caetano_PM10'
N = 10

arrayAccuracy = np.zeros(N)
arrayTime = np.zeros(N)
array_last_Accuracy = np.zeros(N)
        
for i in range(0,N):
    accuracy, compTime, Error, last_accuracy = OSELM(N0, nHiddenNeurons, Block, windowSize, dataSet)
    arrayAccuracy[i] = accuracy
    arrayTime[i] = compTime
    array_last_Accuracy[i] = last_accuracy
    print(i+1)
        
print(" ")
print("OS-ELMsw", " - nHiddenNeurons: ", nHiddenNeurons, " - windowSize: ", windowSize, " - N0: ", N0)
meanAccuracy = np.round(np.mean(arrayAccuracy),5)
meanTime = np.round(np.mean(arrayTime),5)
stdAccuracy = np.round(np.std(arrayAccuracy),5)
stdTime = np.round(np.std(arrayTime),5)
print("Mean Accuracy: ", meanAccuracy, "± ",stdAccuracy)
print("Mean Time: ", meanTime, "± ",stdTime)


fileName = dataSet + "_OS-ELMsw_Standard_Error.csv"
np.savetxt(fileName, Error, delimiter=";", fmt="%s")

A = np.c_[arrayAccuracy, arrayTime]
fileName = dataSet + "_OS-ELMsw_Standard_Output.csv"
np.savetxt(fileName, A, delimiter=";", fmt="%s")

lastAccu = str(np.round(np.mean(array_last_Accuracy),5)) + " ± " + str(np.round(np.std(array_last_Accuracy),5)) 
totalAccu = "Accuracy: " + str(meanAccuracy) + " ± " + str(stdAccuracy) +\
 " windowSize: " + str(windowSize) + " - N0: " + str(N0) + " - nHiddenNeurons: " +\
 str(nHiddenNeurons) + " - Last accuracy: " + lastAccu
fileName = dataSet + "_EOS_Standard_Error.txt"
with open(fileName, "w") as text_file:
    print(f"Mean Accuracy: {totalAccu}", file=text_file)    