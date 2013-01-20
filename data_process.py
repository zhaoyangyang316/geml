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
import AD
import tools

def process():
    print "Process to complete"

def loaddata():
    data = genfromtxt("AP_Uterus_Kidney.csv", delimiter=",")
    data = data[1:,1:]
    data[:,:-1] = log2(data[:,:-1])
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
    print "computing covariace matrix"
    S = ( X.T * X)/ X.shape[0]
    print "computing SVD"
    U, O, V = linalg.svd(S, full_matrices=True)
    O=diag(O[:outDim])**0.5
    V=V[:outDim,:]
    #V=V[:outDim,:]
    return (mat(O)*V*X.T).T



def auto_decoder(dOut, alpha,batch):
#check and load data
    #if(not data_process.check()):
    #data_process.process()
    #data = data_process.loaddata()
    #n_exemple = data.shape[0]
   # d = data.shape[1]-1
   # data[:,:-1]=tools.normalisation(data[:,:-1])
    data = loadtxt("nntest.txt")
    data= normalisation(data[:,:-1])
    print data[0]
    n_exemple = data.shape[0]
    d = data.shape[1]
    #print data
#shuffle the data
    inds = range(n_exemple)
    random.shuffle(inds)
    data = data[inds,:]
#create and train the model
    model = AD.AutoDecoder(dOut,d,alpha,batch)
    model.train(data)
