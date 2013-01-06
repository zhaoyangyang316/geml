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
import numpy
import random
import pylab
import time
import NN
import tools

def process():
    print "Process to complete"

def loaddata():
    data = numpy.loadtxt('iris.txt')
    return data

def check():
    return True

def normalisation():
    print "normalisation to complete"
