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
   
   
   def __init__(self, iw = None, bias = None, m = None, beta = None):      
      self.error = []
      self.errorAcum = float("inf")
      self.IW = iw
      self.Bias = bias
      self.M = m
      self.beta = beta
      self.life = 0
      self.MSE = 0
      self.w = 1
   
   def replaceModels(self, ensemble, worstModel, newModel):
      del ensemble[worstModel]
      ensemble.append(newModel)
      return ensemble
  
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
          self.MSE = (MSE + (e[-1]/windowSize) - (e[0]/windowSize))
#          print("if 3: ", self.MSE, ' life: ', life)

#      if self.MSE == 0:
#          print("MSE: ", self.MSE, " life: ", life, " error[-1]: ", e[-1], " error[0]: ", e[0])
      
   
   def calculateWeight(self, ensemble):
       AveMSE = 0
       MSE = self.MSE
       if len(ensemble) > 1:
           for modl in ensemble:
               AveMSE += modl.MSE
           if AveMSE != 0:
              x = exp(-((MSE - AveMSE)/AveMSE))
              self.w = x
           elif AveMSE == 0:
              x = exp(1)
              print(MSE)
              self.w = x
#       print("w: ", x)
      
   def calculateErrors(self, ensemble, Ptemp, Ttemp):
      errors = []
      for modl in ensemble: 
          Htemp = SigActFun(modl.Bias, Ptemp, modl.IW)
          Ytemp = np.dot(Htemp, modl.beta)
          modl.errorAcum = mean_squared_error(Ttemp, Ytemp)
          errors.append(modl.errorAcum)
          
      worstModel = errors.index(max(errors))
      return worstModel
          
      