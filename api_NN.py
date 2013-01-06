#!/usr/bin/env python
# encoding: utf-8
"""
api_NN.py

Created by YANGYANG ZHAO on 2013-01-03.

"""
# coding=utf-8

###
# Tools
###


import data_process
import numpy
import random
import pylab
import time
import NN
import tools

def run_classify(n_train,n_test,n_valid,alpha,lamda,batch):
#check and load data
    if(not data_process.check()):
        data_process.process()
    data = data_process.loaddata()
    n_exemple = data.shape[0]
    d = data.shape[1]-1
    data[:,:-1]=tools.normalisation(data[:,:-1])

#shuffle the data
    inds = range(n_exemple)
    random.shuffle(inds)

#split data
    tmp_test = n_train+n_test
    tmp_valid = tmp_test + n_valid
    inds_train = inds[:]
    inds_test = inds[n_train:tmp_test]
    inds_valid = inds[tmp_test:tmp_valid]
    train_data = data[inds_train,:]
    print "Train data shape: ",train_data.shape
    test_data = data[inds_test,:]
    valid_data = data[inds_valid,:]
    test_input = test_data[:,:-1]
    test_labels = test_data[:,-1]
    valid_input  = valid_data[:,:-1]
    valid_labels = valid_data[:,-1]

#define param
# Nombre de classes
    n_classes = 3
    m = n_classes

#create and train the model
    model = NN.NN(m,d,alpha,lamda,batch)
    model.train(train_data,valid_input,valid_labels)

#compute the prediction on test data
    t1 = time.clock()
    les_comptes = model.compute_predictions(test_input)
    t2 = time.clock()
    print 'It takes ', t2-t1, ' secondes to compute the prediction on ', test_data.shape[0],' points of test'
    classes_pred = numpy.argmax(les_comptes,axis=1)+1
    confmat = tools.teste(test_labels, classes_pred,n_classes)
    print 'La matrice de confusion est:'
    print confmat

# Error of test
    sum_preds = numpy.sum(confmat)
    sum_correct = numpy.sum(numpy.diag(confmat))
    print "The error of test is ", 100*(1.0 - (float(sum_correct) / sum_preds)),"%"


