# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:11:03 2017

@author: Andres
"""
import numpy as np

##### Methods
def Sigmoide(V):
	return 1/(1 + np.exp(-V))
	
def SigActFun(Bias, Ptemp, IW):
    V = np.dot(Ptemp,np.transpose(IW))
    BiasMatrix = np.repeat(Bias, Ptemp.shape[0], axis=0)
    V =  V + BiasMatrix
    vSigm = np.vectorize(Sigmoide)
    H0 = vSigm(V)
    return H0