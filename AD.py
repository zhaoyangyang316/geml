#!/usr/bin/env python
# encoding: utf-8
"""
NN_m.py

Created by YANGYANG ZHAO on 2013-01-16.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.
"""
import time
import sys
import os
from numpy import *
#from scipy import linalg
import math
#import random
import tools
	
class AutoDecoder:
	def softmax(self,X,axis_inds):
		return exp(X)/sum(exp(X),axis=axis_inds)

	def __init__(self,dh, d, alpha,batch):
		#global m,n,k,W0,W1,B0,B1
		self.alpha = alpha
		self.d=d   #input dimension
		self.dh=dh  #nbre de neurones dans la couche caché
		self.batch=batch #longeur de batch

		self.best_model_perte = 9999.9

		self.W1 = matrix(zeros((self.dh,self.d)))   # W1=mat(dh*d)
		self.W2 = matrix(zeros((self.d,self.dh)))   # W2=mat(m*dh)
		self.aW1 = matrix(zeros((self.dh,self.d)))   # W1=mat(dh*d)
		self.aW2 = matrix(zeros((self.d,self.dh)))   # W2=mat(m*dh)
		self.B1 = matrix(zeros((self.dh,1)))   #b1 = mat(dh*1)
		self.B2 = matrix(zeros((self.d,1)))   #b2 = mat(m*1)
		#self.GaW1 = matrix(zeros((self.dh,self.d)))   # W1=mat(dh*d)
		#self.GaW2 = matrix(zeros((self.m,self.dh)))   # W2=mat(m*dh)
		self.S2 = matrix(zeros((self.d,self.batch)))   # matrix sigma2
		self.S1 = matrix(zeros((self.dh,self.batch)))   # matrix sigma1

		self.G2 = matrix(zeros((self.d,1)))
		self.G1 = matrix(zeros((self.dh,1)))


		self.os = matrix(zeros((self.d,self.batch)))
		self.hs = matrix(zeros((self.dh,self.batch)))


		self.BestW1 = matrix(zeros((self.dh,self.d)))   # W1=mat(dh*d)
		self.BestW2 = matrix(zeros((self.d,self.dh)))   # W2=mat(m*dh)
		self.BestB1 = matrix(zeros((self.dh,1)))   #b1 = mat(dh*1)
		self.BestB2 = matrix(zeros((self.d,1)))   #b2 = mat(m*1)



		print 'la dimension de entree is ',d

		print 'nombre de neurones utilise est ',dh

		#random.seed(8888)
		#initialisation de W1
		interv=1/(d**0.5)
		for i in range(self.dh):   #dh*d
			for j in range(self.d):
				self.W1[i,j] = random.uniform(-interv,interv)

		# TODO 11.28
		#initialisation de W2
		#random.seed(3646)
		interv=1/(dh**0.5)
		for i in range(self.d):   #m*dh
			for j in range(self.dh):
				self.W2[i,j] = random.uniform(-interv,interv)
		print 'initialisation fini'
		#print self.W1
		#print self.W2


		#self.W1=mat(random.uniform((-1.0/sqrt(self.dh)),(1.0/sqrt(self.dh)),(d*self.dh)).reshape(self.dh,d))
		#self.W2=mat(random.uniform((-1.0/sqrt(self.m)),(1.0/sqrt(self.m)),(self.m*self.dh)).reshape(self.m,self.dh))

		

	def calculate_forward(self,X):   #calcule hs et os
		#X sous forme de d*1
		#print X
		#print self.W1
		self.hs = tanh(self.W1*X.T+self.B1)
		self.os = self.W2 * self.hs + self.B2


	def calculate_backward(self,X):
		#sigma2 d(C)/d(oa)
		self.S2 = 2*(self.os - X.T)
		#print self.os

		#sigma1 d(C)/d(ha)
		#delta_hs = matrix(zeros((self.dh,batch)))
		self.S1 = self.W2.T * self.S2
		self.S1 = mat(self.S1.A * (1-(self.hs.A)**2))
		#print 
		self.aW1=self.S1 * X 
		self.aW2=self.S2 * self.hs.T 
		#print "fdaf",self.W1
		

	def adjust_weight(self):
		#X est 1*d
		self.W2 = self.W2 - self.alpha * self.aW2 - 0.05* self.W2
		self.W1 = self.W1 - self.alpha * self.aW1 - 0.05* self.W1
		self.B2 = self.B2 - self.alpha * self.G2
		self.B1 = self.B1 - self.alpha * self.G1
		

	def train(self, train_set):
		#Be sure the data are normalized


		
		changeTimer = 0
		print 'training rate is ',self.alpha
		finish = False
		nb_train=train_set.shape[0]
		print 'training set size is ',nb_train
		max_iter = 20*nb_train
		itera=0
		seuil = 0.0
		epoch = 0
		perte = 999999.9
		
		#k=0
		X=mat(train_set)
		#print X
		print max_iter;

		#record = []
		#filename="output"

		while not finish:
			inds = range(train_set.shape[0])
			random.shuffle(inds)
			self.calculate_forward(X[inds[:self.batch],:])
			self.calculate_backward(X[inds[:self.batch],:])
	
			itera+=1
				
			self.G1=sum(self.S1,axis = 1)
			self.G2=sum(self.S2,axis = 1)


			self.adjust_weight()

			epoch =  itera*self.batch / nb_train 

			if ((itera*self.batch)%nb_train == 0):
				print "Epoch ",epoch
				#perte
				les_comptes=self.compute_predictions(train_set)
				perte =self.compute_loss(les_comptes,X)
				print "Epoch ", epoch,": perte:",perte,", computing time is ",time.clock()," seconds"
				print self.os

				if (perte < self.best_model_perte):
					self.best_model_perte = perte
					
					self.BestB1 = self.B1
					self.BestW1 = self.W1
					self.BestB2 = self.B2					
					self.BestW2 = self.W2
					print "Best model until now with perte :",self.best_model_perte,", computing time is ",time.clock()," seconds"
					
				#record.append([taux_train,self.best_model_valid_error,self.best_model_test_error,perte_train,perte_valid,perte_test])
				#savetxt(filename,array(record),delimiter=" ")



			if (perte< seuil) or (itera > max_iter):
				finish = True
	
		
		print "W1:"
		print self.BestW1
		print "W2"
		print self.BestW2

	def compute_predictions(self,testData):
		nb_test=testData.shape[0]
		sorties = mat(zeros((nb_test,self.d)))
		for i in range(nb_test):
			data=mat(testData[i,:]).T
			HS=tanh(self.W1 * data + self.B1)
			sorties[i,:]=(self.W2*HS +self.B2).T
			#print sorties[i,:]
		return sorties.A

	def compute_loss(self,pred,X):
		print pred.shape
		print pred[0]
		print X.shape
		delta = ((pred - X)/X.shape[0]).A
		perte = sum((delta**2)**0.5) /X.shape[1]
		return perte