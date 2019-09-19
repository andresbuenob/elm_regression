# -*- coding: utf-8 -*-
"""
Created on Sat May 27 10:08:21 2017

@author: Andres
"""
import numpy as np
from ActivationFunction import SigActFun
from sklearn.metrics import mean_squared_error
from math import exp

class Model:
   'Common base class for all employees'
   
   def __init__(self, iw = None, bias = None, m = None, beta = None):      
      self.error = []
      self.IW = iw
      self.Bias = bias
      self.M = m
      self.beta = beta
      self.life = 0
      self.MSE = 0
      self.w = 1
   
   def replaceModels(self, ensemble, newModel):
      errors = []
      for modl in ensemble: 
          errors.append(modl.MSE)
      worstModel = errors.index(max(errors))
      del ensemble[worstModel]
      ensemble.append(newModel)
      return ensemble
  
   def rank(self, ensemble):
      ensemble.sort(key=lambda x: x.MSE)
#      for modl in ensemble:
#          print(modl.MSE)
  
   def calculateError(self, modelOutput, Real, windowSize):
      e =  np.power((Real-modelOutput), 2)
      self.life += 1
      if len(self.error) < windowSize:
          self.error.append(e)
      else:
         del self.error[0]
         self.error.append(e)
    
               
   def calculateMSE(self, windowSize):
      life = self.life
      MSE = self.MSE
      e = self.error 
      if life == 0:
          self.MSE = 0
#          print("if 1: ", self.MSE, ' life: ', life)
      elif life >= 1 and life <= windowSize:
          self.MSE = (((life-1)/life) * MSE + (1/life) * e[-1])
#          print("if 2: ", self.MSE, ' life: ', life)
      elif life > windowSize:
          self.MSE = (MSE + e[-1]/windowSize - e[0]/windowSize)
#          print("if 3: ", self.MSE, ' life: ', life)
      
   
   def calculateWeight(self, ensemble):
       AveMSE = 0
       MSE = self.MSE
       medianMSE = []
       if len(ensemble) > 1:
           for modl in ensemble:
               AveMSE += modl.MSE
               medianMSE.append(modl.MSE)
           medMSE = np.median(medianMSE)
           if medMSE != 0:
              x = exp(-((MSE - medMSE)/medMSE))
              self.w = x
           elif medMSE == 0:
              x = exp(0)
              self.w = x
#       print("w: ", x)
              
      
   def calculateErrors(self, ensemble, Ptemp, Ttemp):
      errors = []
      for modl in ensemble: 
          Htemp = SigActFun(modl.Bias, Ptemp, modl.IW)
          Ytemp = np.dot(Htemp, modl.beta)
          modl.error = mean_squared_error(Ttemp, Ytemp)
          errors.append(modl.error)
          
      worstModel = errors.index(max(errors))
      return worstModel
          
      