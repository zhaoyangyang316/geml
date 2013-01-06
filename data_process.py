#!/usr/bin/env python
# encoding: utf-8
"""
data.py

Created by YANGYANG ZHAO on 2013-01-03.

"""
# coding=utf-8

###
# Tools
###
from numpy import *
import random
import pylab
import time
import NN
import tools

def process():
    print "Process to complete"

def loaddata():
    data = loadtxt('iris.txt')
    return data

def check():
    return True

def normalisation(X):
    nb=X.shape[0]
    mu= sum(X,axis=0)/nb
    out=X-mu
    ecart= (sum(out**2,axis=0)/nb)**0.5
    for i in range(X.shape[1]):
        if ecart[i]!=0:
            out[:,i]=out[:,i]/ecart[i]
    return out

def pca(data,outDim):
    X = mat(normalisation(data))
    # compute the covariace matrix
    S = ( X.T * X)/ X.shape[0]
    U, O, V = linalg.svd(S, full_matrices=True)
    O=diag(O[:outDim])**0.5
    V=V[:outDim,:]
    #V=V[:outDim,:]
    return (mat(O)*V*X.T).T