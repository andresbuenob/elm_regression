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
import random

##### Functions
def trainModel(nHiddenNeurons, nInputNeurons, P0, T0):
    iw = np.random.uniform(-1, 1, (nHiddenNeurons,nInputNeurons))
    bias = np.random.uniform(-1, 1, (1,nHiddenNeurons))
    H0 = SigActFun(bias, P0, iw)
    m = np.linalg.pinv(np.dot(np.transpose(H0),H0))
    beta = np.dot(np.linalg.pinv(H0),T0)
    model = Model(iw, bias, m, beta)
    return model

def EOS(N0, nHiddenNeurons, Block, nModels, trBlock, maxWindowSize, dataset):
    modelObj = Model()
    ##### Load dataset
    try:
        data = np.genfromtxt(dataset, dtype = float)
    except Exception:
        print('An error ecurred while loading data')
    
    T = np.array(data[:,0])
    P = np.array(data[:,1:data.shape[1]])
    
    nInputNeurons = data.shape[1]-1
    nTrainingData = data.shape[0]                          
    
    ##### Step 1 Initialization Phase
    P0 = P[0:N0,:]
    T0 = T[0:N0]
    
    PT = np.array(P[N0-maxWindowSize:nTrainingData,:])
    TT = np.array(T[N0-maxWindowSize:nTrainingData])
    Real = np.array(T[N0-1:nTrainingData-1])
    Y = np.zeros_like(Real)
    Error = np.zeros_like(Real)
    
    start_time = time()
    errorSum = 0
    ensemble = []
    ensemOutput = []
	
    for n in range(1, 11):
        b = random.randrange(maxWindowSize, N0)
        P0 = np.array(P[(N0 - b):N0])
        T0 = np.array(T[(N0 - b):N0])
        ensemble.append(trainModel(nHiddenNeurons, nInputNeurons, P0, T0))

    nModelsCounter = 0
    ##### Step 2 Sequential Learning Phase
    for n in range(maxWindowSize, PT.shape[0]):
        windowSize = maxWindowSize
        Ptemp1 = np.array(PT[(n-windowSize):n])
        Ttemp1 = np.array(TT[(n-windowSize):n])
		
        for modl in ensemble:   
            beta = modl.beta
            IW = modl.IW
            Bias = modl.Bias
            M = modl.M
            H = SigActFun(Bias, Ptemp1, IW)
            
            g = np.dot(H,beta) 
            
            if nModelsCounter < 10:
                ensemOutput.append(g[-1])
                nModelsCounter += 1
			
            modl.calculateError(g[-1],Ttemp1[-1], trBlock)  
            modl.calculateMSE(trBlock)            
			
            K = np.dot(M,np.transpose(H))
            Q = np.linalg.inv(np.eye(windowSize) + np.dot(H, K))
            R = np.dot(K, Q)
            S = np.dot(H, M)
            M = M - np.dot(R, S)
            
            beta = beta + np.dot(np.dot(M,np.transpose(H)),(Ttemp1-np.dot(H,beta)))
            modl.beta = beta
            modl.M = M
        
        nModelsCounter = 0   
        modelObj.rank(ensemble)
        Y[n-maxWindowSize] = np.average(ensemOutput)
        ensemOutput = []
        
        errorSum += np.power((Ttemp1[-1]-Y[n-maxWindowSize]), 2)
        Error[n-maxWindowSize] = errorSum/((n-maxWindowSize)+1)
					
        if n % trBlock == 0 and n != 0:
            
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
    
    last_Real = Real[-200:]
    last_Y = Y[-200:]
    last_accuracy = mean_squared_error(last_Real, last_Y)
    
    return accuracy, compTime, Error, last_accuracy

##### Macro definition
N0 = 120
nHiddenNeurons = 15
#amount of samples used to update de model in each step
Block = 1
nModels = 20
#training block of samples for new models 
trBlock = 60
windowSize = 3
maxWindowSize = 3
N = 10
dataset = 'Sao_Caetano_PM10'

arrayAccuracy = np.zeros(N)
arrayTime = np.zeros(N)
array_last_Accuracy = np.zeros(N)


for i in range(0,N):
    accuracy, compTime, Error, last_accuracy = EOS(N0, nHiddenNeurons, Block, nModels, trBlock, maxWindowSize, dataset)
    arrayAccuracy[i] = accuracy
    arrayTime[i] = compTime
    print(compTime)
    array_last_Accuracy[i] = last_accuracy
    print(i+1)
             
print(" ")
print("EOS", " - maxWindowSize: ", maxWindowSize, " - N0: ", N0, " - trBlock: ", trBlock, " - nModels: ", nModels)
meanAccuracy = np.round(np.mean(arrayAccuracy),5)
meanTime = np.round(np.mean(arrayTime),5)
stdAccuracy = np.round(np.std(arrayAccuracy),5)
stdTime = np.round(np.std(arrayTime),5)
print("Mean Accuracy: ", meanAccuracy, "± ",stdAccuracy)
print("Mean Time: ", meanTime, "± ",stdTime)

fileName = dataset + "_EOS_Rank_Error.csv"
np.savetxt(fileName, Error, delimiter=";", fmt="%s")

A = np.c_[arrayAccuracy, arrayTime]
fileName = dataset + "_EOS_Rank_Output.csv"
np.savetxt(fileName, A, delimiter=";", fmt="%s")

lastAccu = str(np.round(np.mean(array_last_Accuracy),5)) + " ± " + str(np.round(np.std(array_last_Accuracy),5))
totalAccu = "Accuracy: " + str(meanAccuracy) + " ± " + str(stdAccuracy) + " windowSize: " + str(windowSize) + " - N0: " + str(N0) + " - trBlock: " +  str(trBlock) + " - nModels: " + str(nModels) + " - Last accuracy: " + lastAccu
fileName = dataset + "_EOS_Rank_Error.txt"
with open(fileName, "w") as text_file:
    print(f"Mean Accuracy: {totalAccu}", file=text_file)