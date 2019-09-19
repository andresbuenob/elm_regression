"""
Created on Tue May 23 13:35:52 2017

@author: Andres
"""
#encoding utf-8
import numpy as np
from time import time
from sklearn.metrics import mean_squared_error
from ActivationFunction import SigActFun
from Model import Model

##### Functions
def trainModel(nHiddenNeurons, nInputNeurons, P0, T0):
    iw = np.random.uniform(-1, 1, (nHiddenNeurons,nInputNeurons))
    bias = np.random.uniform(-1, 1, (1,nHiddenNeurons))
    H0 = SigActFun(bias, P0, iw)
    m = np.linalg.pinv(np.dot(np.transpose(H0),H0))
    beta = np.dot(np.linalg.pinv(H0),T0)
    model = Model(iw, bias, m, beta)
    return model

def EOS(N0, nHiddenNeurons, Block, nModels, trBlock, windowSize, dataSet):
    modelObj = Model()
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
    
    ensemble = []
    ensemOutput = []
    
    
    ensemble.append(trainModel(nHiddenNeurons, nInputNeurons, P0, T0))
    
    ##### Step 2 Sequential Learning Phase
    for n in range(windowSize, PT.shape[0]):  
        Ptemp1 = np.array(PT[(n-windowSize):n])
        Ttemp1 = np.array(TT[(n-windowSize):n])
        i = 0
        for modl in ensemble:   
            beta = modl.beta
            IW = modl.IW
            Bias = modl.Bias
            M = modl.M
            H = SigActFun(Bias, Ptemp1, IW)
            
            g = np.dot(H,beta) 
            ensemOutput.append(g[windowSize-1])
#            e = modl.calculateError(g[windowSize-1],Ttemp1[windowSize-1], windowSize)
#            totalErrors[i,n-windowSize] = e
#            i += 1
            
            K = np.dot(M,np.transpose(H))
            Q = np.linalg.inv(np.eye(windowSize) + np.dot(H, K))
            R = np.dot(K, Q)
            S = np.dot(H, M)
            M = M - np.dot(R, S)
            
            beta = beta + np.dot(np.dot(M,np.transpose(H)),(Ttemp1-np.dot(H,beta)))
            modl.beta = beta
            modl.M = M
            
        Y[n-windowSize] = np.average(ensemOutput)
        ensemOutput = []
        
        if n % trBlock == 0 and n != 0 :
            Ptemp2 = np.array(PT[(n - trBlock):n])
            Ttemp2 = np.array(TT[(n - trBlock):n])
            newModel = trainModel(nHiddenNeurons, nInputNeurons, Ptemp2, Ttemp2)
            worstModel = modelObj.calculateErrors(ensemble, Ptemp2, Ttemp2, i)
            if len(ensemble) < nModels:   
                ensemble.append(newModel)
            else:
                ensemble = modelObj.replaceModels(ensemble, worstModel, newModel)
                
    
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
N0 = 120
nHiddenNeurons = 5
#amount of samples used to update de model in each step
Block = 1
nModels = 10
#training block of samples for new models 
trBlock = 100
windowSize = 3
N = 5
dataSet = 'Sao_Caetano_PM10'

arrayAccuracy = np.zeros(N)
arrayTime = np.zeros(N)
trBlocks = np.array([10, 20, 30, 40, 50, 60, 80, 100, 120, 200])
windows = np.array([3, 5, 10])
hiddenNeurons = np.array([5, 10, 15])
#accuracyOutput = np.zeros(len(hiddenNeurons))
accuracyOutput = []
A = trBlocks

#for j in range(0, len(windows)):
#    windowSize = windows[j]
#    accuracyOutput = []
for j in range(0, len(trBlocks)):
    trBlock = trBlocks[j]
    
    for i in range(0,N):
        accuracy, compTime = EOS(N0, nHiddenNeurons, Block, nModels, trBlock, windowSize, dataSet)
        arrayAccuracy[i] = accuracy
        arrayTime[i] = compTime
        print(i+1)       
        
    print(" ")
    print("EOS", " - windowSize: ", windowSize, " - N0: ", N0, " - trBlock: ", trBlock, " - nModels: ", nModels, " Hidden Neurons: ", nHiddenNeurons)
    meanAccuracy = np.round(np.mean(arrayAccuracy),5)
    meanTime = np.round(np.mean(arrayTime),5)
    stdAccuracy = np.round(np.std(arrayAccuracy),5)
    stdTime = np.round(np.std(arrayTime),5)
    print("Mean Accuracy: ", meanAccuracy, "± ",stdAccuracy)
    print("Mean Time: ", meanTime, "± ",stdTime)
    accuracyAndStd = str(meanAccuracy)
    accuracyOutput.append(accuracyAndStd)

A = np.c_[A, accuracyOutput]
#    A = np.concatenate((A, accuracyOutput))

#A = np.column_stack((Y,Real))
#f = open('exampleData.csv', 'wb')
#np.savetxt(f, A, delimiter=";", fmt="%s")
fileName = dataSet + "_EOS_output.csv"
np.savetxt(fileName, A, delimiter=";", fmt="%s")