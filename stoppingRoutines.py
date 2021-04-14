#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 14:41:46 2018

define a class for handling stopping criteria in training

@author: claudio
"""

import numpy as np

class Stopper:
    max_error=1e6
    def __init__(self,striplength):
        self.k=striplength
        self.errors=self.max_error*np.ones([self.k,1])
        self.minimal_training_error=self.max_error
        self.minimal_validation_error=self.max_error
        self.GL=self.max_error

    def insert(self,tmp):
        #insert a new value int error
        tmpv=np.zeros((self.k,1))
        tmpv[-1]=tmp        
        for i1 in range(self.k-1):
            tmpv[i1]=self.errors[i1+1]
        self.errors=tmpv
        # update minima training error
        if tmp<self.minimal_training_error:
            self.minimal_training_error=tmp
            
    def generalization_loss(self,tmp):
        # return the generalization loss and update the minimal validation error
                   # evaluate generalization loss 
        self.GL=100.0*(-1.0+tmp/self.minimal_validation_error)
        if tmp<self.minimal_validation_error:
            self.minimal_validation_error=tmp        
            
        return self.GL
        
    def progress(self):
        # return the "progress"
        tmp=np.sum(self.errors)
        self.Pk=1000*(-1.0+tmp/(self.k*np.min(self.errors)))
        return self.Pk

    def quotient(self):
        # return the quotients of the generalized loss and of the progress
        # use the last stored values
        return self.GL/self.Pk