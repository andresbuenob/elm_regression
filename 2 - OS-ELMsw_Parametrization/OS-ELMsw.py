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
    
    start_time = time()
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
    
    ##### Time in seconds
    end_time = time()
    totalTime = end_time - start_time
    hours, rest = divmod(totalTime,3600)
    minutes, seconds = divmod(rest, 60)
    compTime = np.round(seconds,5)
    
    ##### Acurracy MSE
    mse = mean_squared_error(Real, Y)
    accuracy = np.round(mse,5)
    
    return accuracy, compTime


##### Macro definition
N0 = 90
nHiddenNeurons = 5
windowSize = 3
Block = 1
dataSet = 'Sao_Caetano_PM10'
N = 10

arrayAccuracy = np.zeros(N)
arrayTime = np.zeros(N)
windows = np.array([3, 5, 10, 20, 30, 60, 90])
hiddenNeurons = np.array([5, 10, 15, 20, 25])
#accuracyOutput = np.zeros(len(hiddenNeurons))
accuracyOutput = []
A = hiddenNeurons

for j in range(0, len(windows)):
    windowSize = windows[j]
    accuracyOutput = []
    for j in range(0, len(hiddenNeurons)):
        nHiddenNeurons = hiddenNeurons[j]
        
        for i in range(0,N):
            accuracy, compTime = OSELM(N0, nHiddenNeurons, Block, windowSize, dataSet)
            arrayAccuracy[i] = accuracy
            arrayTime[i] = compTime
            print(i+1)
        
        print(" ")
        print("OS-ELMsw", " - windowSize: ", windowSize, " - Hidden Neurons: ", nHiddenNeurons, " - N0: ", N0)
        meanAccuracy = np.round(np.mean(arrayAccuracy),5)
        meanTime = np.round(np.mean(arrayTime),5)
        stdAccuracy = np.round(np.std(arrayAccuracy),5)
        stdTime = np.round(np.std(arrayTime),5)
        print("Mean Accuracy: ", meanAccuracy, "± ",stdAccuracy)
        print("Mean Time: ", meanTime, "± ",stdTime)
#        accuracyAndStd = str(meanAccuracy) + " ± " + str(stdAccuracy)
        accuracyAndStd = str(meanAccuracy)
        accuracyOutput.append(accuracyAndStd)
    
    A = np.c_[A, accuracyOutput]

fileName = dataSet + "_OS-ELMsw_output.csv"
np.savetxt(fileName, A, delimiter=";", fmt="%s")