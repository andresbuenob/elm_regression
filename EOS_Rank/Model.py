# -*- coding: utf-8 -*-
"""
Created on Sat May 27 10:08:21 2017

@author: Andres
"""
import numpy as np
from ActivationFunction import SigActFun
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class Model:
   'Common base class for all employees'
   modelsCount = 0
   
   def __init__(self, iw = None, bias = None, m = None, beta = None):      
      self.error = []
      self.IW = iw
      self.Bias = bias
      self.M = m
      self.beta = beta
      self.life = 0
      self.MSE = 0
      self.w = 1
   
   
   def replaceModels(self, ensemble, worstModel, newModel):
#      errors = []
#      for modl in ensemble: 
#          e = modl.MSE
#          errors.append(e)
#      worstModel = errors.index(max(errors))
      ensemble[worstModel] = newModel
      return ensemble
  
   def rank(self, ensemble):
      ensemble.sort(key=lambda x: x.MSE)
#      for modl in ensemble:
#          print(modl.MSE)
      
   
   def calculateErrors(self, ensemble, Ptemp, Ttemp, n):
      errors = []
      for modl in ensemble: 
          Htemp = SigActFun(modl.Bias, Ptemp, modl.IW)
          Ytemp = np.dot(Htemp, modl.beta)
          e = mean_squared_error(Ttemp, Ytemp)
          errors.append(e)
      worstModel = errors.index(max(errors))
      return worstModel
  
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

   def plotError(self, Error):
       ##### Graphics
        fig, axes = plt.subplots(figsize=(16,7))
        x = np.arange(len(Error))
        axes.plot(x, Error, 'black', label='Mean Squared Error')
#        axes.plot(x, Real, 'red', label='Real data')
        plt.xlabel('Samples')
        plt.ylabel('MSE')
        axes.legend(loc='upper right')
        axes.set_title('Online Error')
#        axes.set_ylim([0,1.5])
        #fig.canvas.manager.window.wm_geometry("+%d+%d" % (20, 20))
        plt.show()
          
      