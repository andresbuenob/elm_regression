# -*- coding: utf-8 -*-
"""
Created on Sat May 27 10:08:21 2017

@author: Andres
"""
import numpy as np
from ActivationFunction import SigActFun
from sklearn.metrics import mean_squared_error

class Model:
   'Common base class for all employees'
   modelsCount = 0
   
   def __init__(self, iw = None, bias = None, m = None, beta = None):      
      self.error = float("inf")
      self.errores = np.zeros(56)
      self.IW = iw
      self.Bias = bias
      self.M = m
      self.beta = beta
      Model.modelsCount += 1
   
   def replaceModels(self, ensemble, worstModel, newModel):
      ensemble[worstModel] = newModel
      return ensemble
      
      
   def calculateErrors(self, ensemble, Ptemp, Ttemp, n):
      errors = []
      i = 0
      for modl in ensemble: 
          Htemp = SigActFun(modl.Bias, Ptemp, modl.IW)
          Ytemp = np.dot(Htemp, modl.beta)
          e = mean_squared_error(Ttemp, Ytemp)
          errors.append(e)
          modl.errores[n] = e
#          totalErrors[i,n] = e
          i += 1
#          modl.errores.append(mean_squared_error(Ttemp, Ytemp))
#      totalErrors[10,n] = np.mean(errors)
      worstModel = errors.index(max(errors))
      return worstModel
  
   def calculateError(self, modelOutput, Real, windowSize):
      e =  np.power((Real-modelOutput), 2)
      return e
          
      