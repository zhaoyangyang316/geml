#!/usr/bin/env python
# encoding: utf-8
"""
NN.py

Created by YANGYANG ZHAO on 2013-01-03.
"""
import time
import sys
import os
from numpy import *
#from scipy import linalg
import math
import utilitaires

class NN:
    def softmax(self,X):
        return exp(X)/sum(exp(X))

    def sigmoid(self,X):
        return 1/(1+e**(-X))

	def __init__(self, m,d, alpha,lamda,batch,NN_type = 0):
		self.alpha = alpha
		self.m= m  #output dimension
		self.d=d   #input dimension
		self.dh=dh  #nbre de neurones dans la couche caché
		self.batch=batch #longeur de batch
		self.lamda=lamda

        print "Initialize the NN model."

		self.W1 = matrix(zeros((self.dh,self.d)))   # W1=mat(dh*d)
		self.aW1 = matrix(zeros((self.dh,self.d)))   # W1=mat(dh*d)
		self.B1 = matrix(zeros((self.dh,1)))   #b1 = mat(dh*1)
		self.GaW1 = matrix(zeros((self.dh,self.d)))   # W1=mat(dh*d)
		self.S1 = matrix(zeros((self.dh,1)))   # matrix sigma1
		self.G1 = matrix(zeros((self.dh,1)))
		self.os = matrix(zeros((self.m,1)))

        print "The dimension of input is ",d
        print "The dimension of output is ",m

		#initialisation de W1
		interv=1/(d**0.5)
		for i in range(self.dh):   #dh*d
			for j in range(self.d):
				self.W1[i,j] = random.uniform(-interv,interv)

		print 'initialisation fini'

	def calculate_forward(self,X):   #calcule hs et os
        if (NN_type == 0):
            self.os = self.softmax(self.W1 * X + self.B1)
        else if(NN_type == 1):
            self.os = self.sigmoid(self.W1 * X + self.B1)
        else:
            print "NN type unknown!"

#TODO here
	def calculate_backward(self,X):
		#sigma2 d(C)/d(oa)
		#T est nu de classe
		#print X
		for i in range(self.m):
			if (i == (X[0,-1]-1)):
				#print self.os[i,0]
				self.S2[i,0]= self.os[i,0] -1
			else:
				self.S2[i,0]= self.os[i,0]

		#sigma1 d(C)/d(ha)
		for i in range(self.dh):
			self.S1[i,0] = sum(self.S2.T * self.W2[:,i]) * (1-(self.hs[i,0])**2)

		self.aW1=self.S1 * X[0,:-1] + self.lamda * self.W1
		self.aW2=self.S2 * self.hs.T + self.lamda*self.W2


	def calculate_forward_matrix(self,X):   #calcule hs et os
		#X sous forme de d*1
		#print X
		#print self.W1
		self.hs = tanh(self.W1*X.T+self.B1)
		self.os = self.softmax(self.W2 * self.hs + self.B2)




	def calculate_backward_matrix(self,X):
		#sigma2 d(C)/d(oa)
		#T est nu de classe
		#print X
		for i in range(self.m):
			if (i == (X[0,-1]-1)):
				#print self.os[i,0]
				self.S2[i,0]= self.os[i,0] -1
			else:
				self.S2[i,0]= self.os[i,0]

		#sigma1 d(C)/d(ha)
		for i in range(self.dh):
			self.S1[i,0] = sum(self.S2.T * self.W2[:,i]) * (1-(self.hs[i,0])**2)

		self.aW1=self.S1 * X[0,:-1] + self.lamda * self.W1
		self.aW2=self.S2 * self.hs.T + self.lamda*self.W2

	def adjust_weight(self):
		#X est 1*d
		self.W2 = self.W2 - self.alpha * self.GaW2
		self.W1 = self.W1 - self.alpha * self.GaW1
		self.B2 = self.B2 - self.alpha * self.G2
		self.B1 = self.B1 - self.alpha * self.G1

	def train(self,train_set,valide_set,valide_labels):
		nb_valide=valide_set.shape[0]
		changeTimer = 0
		print 'training rate is ',self.alpha
		finish = False
		nb_train=train_set.shape[0]
		max_iter = 201*nb_train
		itera=0
		seuil = 1
		taux = 100.0
		taux_valid =100.0
		k=0
		X=mat(train_set)
		#print X
		print max_iter;
		while not finish:
			inds = range(train_set.shape[0])
			random.shuffle(inds)
			for i in range(self.batch):
				self.calculate_forward(X[inds[i],:-1])
				self.calculate_backward(X[inds[i],:])
				#print 'S1 =',self.S1
				#print 'S2 =',self.S2
				self.G1+=self.S1
				self.G2+=self.S2
				self.GaW1+=self.aW1
				self.GaW2+=self.aW2
				itera+=1
				k+=1
				k=k%nb_train
			self.G1=self.G1
			self.G2=self.G2
			self.GaW1=self.GaW1
			self.GaW2=self.GaW2

			self.adjust_weight()


			#taux de erreur pour validation
			#les_comptes=self.compute_predictions(valide_set)
			#classes_pred = argmax(les_comptes,axis=1)+1
			#confmat = utilitaires.teste(valide_labels, classes_pred,self.m)
			#sum_preds = sum(confmat)
			#sum_correct = sum(diag(confmat))
			#taux= 100*(1.0 - (float(sum_correct) / sum_preds))
			#print "L'erreur de test est de validation est ", taux,"%",time.clock()



			if (itera%nb_train == 0):
				#taux de erreur pour validation
				les_comptes=self.compute_predictions(valide_set)
				classes_pred = argmax(les_comptes,axis=1)+1
				confmat = utilitaires.teste(valide_labels, classes_pred,self.m)
				sum_preds = sum(confmat)
				sum_correct = sum(diag(confmat))
				taux_valid= 100*(1.0 - (float(sum_correct) / sum_preds))
				print "L'erreur de test est de validation est ", taux_valid,"%",time.clock()
				#taux de erreur pour train_set
				les_comptes=self.compute_predictions(train_set[:,:-1])
				classes_pred = argmax(les_comptes,axis=1)+1
				confmat = utilitaires.teste(train_set[:,-1], classes_pred,self.m)
				sum_preds = sum(confmat)
				sum_correct = sum(diag(confmat))
				taux= 100*(1.0 - (float(sum_correct) / sum_preds))
				#print "iteration :", itera, ". L'erreur de test est de training est ", taux,"%",time.clock()
				#print "iteration :", itera
				print "L'erreur de test est de training est ", taux,"%",time.clock()


			self.G2 = self.G2*0.0
			self.G1 = self.G1*0.0
			self.GaW1 =self.GaW1*0.0
			self.GaW2 = self.GaW2*0.0



			if (taux < 30) and (changeTimer == 0):
				self.alpha=self.alpha/2
				changeTimer += 1
				print "alpha changed !!! because taux < 30"
				print self.alpha



			if (taux < 10) and (changeTimer == 1):
				self.alpha=self.alpha/5
				changeTimer += 1
				print "alpha changed !!! because taux < 10"
				print self.alpha


			if (taux_valid< seuil) or (itera > max_iter):
				finish = True



		print self.W1
		print self.W2

	def compute_predictions(self,testData):
		nb_test=testData.shape[0]
		sorties = mat(zeros((nb_test,self.m)))
		for i in range(nb_test):
			data=mat(testData[i,:]).T
			HS=tanh(self.W1 * data + self.B1)
			sorties[i,:]=(self.softmax(self.W2*HS +self.B2)).T
		return sorties.A

	def testGrad(self,train_set,x,y):
		X=mat(train_set)
		e=0.0001
		for i in range(train_set.shape[0]):
			self.calculate_forward(X[i,:-1])
			perte=-log(self.os[X[i,-1]-1,0])
			self.calculate_backward(X[i,:])
			#print self.W1[0,1]
			self.W1[x,y]+=e
			#print self.W1[0,1]
			#self.W2+=e
		    #self.B1[x,0]+=e
			#self.B2+=e
			self.calculate_forward(X[i,:-1])
			perte1=-log(self.os[X[i,-1]-1,0])
			delta=(perte1-perte)/e

			print (self.aW1[x,y]/delta)
			#print delta

	def testGradW1(self,train_set):
		X=mat(train_set)
		seuil = 0.01
		e=0.0001
		wrong = False
		perte = 0.0
		perte1 = 0.0
		sumGrad = 0.0
		for x in range(self.W1.shape[0]):
			for y in range(self.W1.shape[1]):
				for i in range(X.shape[0]):
					self.calculate_forward(X[i,:-1])
					perte=-log(self.os[X[i,-1]-1,0]) +perte
					self.calculate_backward(X[i,:])
					#print self.W1[0,1]
					self.W1[x,y]+=e
					#print self.W1[0,1]
					#self.W2+=e
				    #self.B1[x,0]+=e
					#self.B2+=e
					self.calculate_forward(X[i,:-1])
					perte1=-log(self.os[X[i,-1]-1,0]) + perte1
					sumGrad += self.aW1[x,y]
				delta=(perte1-perte)/e

				if (sumGrad/delta > 1+seuil) or (sumGrad/delta < 1-seuil):
					wrong = True
					print sumGrad/delta
					#print delta
				perte = 0.0
				perte1 = 0.0
				sumGrad = 0.0
		if not (wrong):
			print "Grandiant of W1 is checked"
		else:
			print "Grandiant of W1 is wrong"

	def testGradW2(self,train_set):
		X=mat(train_set)
		seuil = 0.01
		e=0.0001
		wrong = False
		perte = 0.0
		perte1 = 0.0
		sumGrad = 0.0
		for x in range(self.W2.shape[0]):
			for y in range(self.W2.shape[1]):
				for i in range(X.shape[0]):
					self.calculate_forward(X[i,:-1])
					perte=-log(self.os[X[i,-1]-1,0]) + perte
					self.calculate_backward(X[i,:])
					#print self.W1[0,1]
					self.W2[x,y]+=e
					#print self.W1[0,1]
					#self.W2+=e
				    #self.B1[x,0]+=e
					#self.B2+=e
					self.calculate_forward(X[i,:-1])
					perte1=-log(self.os[X[i,-1]-1,0]) + perte1
					sumGrad += self.aW2[x,y]
				delta=(perte1-perte)/e
				if (sumGrad/delta > 1+seuil) or (sumGrad/delta < 1-seuil):
					wrong = True
				perte = 0.0
				perte1 = 0.0
				sumGrad = 0.0
		if not (wrong):
			print "Grandiant of W2 is checked"
		else:
			print "Grandiant of W2 is wrong"
			#print delta

	def testGradB1(self,train_set):
		X=mat(train_set)
		seuil = 0.01
		e=0.0001
		wrong = False
		perte = 0.0
		perte1 = 0.0
		sumGrad = 0.0
		for x in range(self.B1.shape[0]):
			for i in range(X.shape[0]):
				self.calculate_forward(X[i,:-1])
				perte=-log(self.os[X[i,-1]-1,0])+perte
				self.calculate_backward(X[i,:])
				#print self.W1[0,1]
				self.B1[x,0]+=e
				#print self.W1[0,1]
				#self.W2+=e
			    #self.B1[x,0]+=e
				#self.B2+=e
				self.calculate_forward(X[i,:-1])
				perte1=-log(self.os[X[i,-1]-1,0])+perte1
				sumGrad += self.S1[x,0]
			delta=(perte1-perte)/e
			if (sumGrad/delta > 1+seuil) or (sumGrad/delta < 1-seuil):
				wrong = True
				print sumGrad
				print delta
			perte = 0.0
			perte1 = 0.0
			sumGrad = 0.0
		if not (wrong):
			print "Grandiant of B1 is checked"
		else:
			print "Grandiant of B1 is wrong"
		#print delta


	def testGradB2(self,train_set):
		X=mat(train_set)
		seuil = 0.01
		e=0.0001
		wrong = False
		perte = 0.0
		perte1 = 0.0
		sumGrad = 0.0
		for x in range(self.B2.shape[0]):
			for i in range(X.shape[0]):
				self.calculate_forward(X[i,:-1])
				perte=-log(self.os[X[i,-1]-1,0])+perte
				self.calculate_backward(X[i,:])
				#print self.W1[0,1]
				self.B2[x,0]+=e
				#print self.W1[0,1]
				#self.W2+=e
			    #self.B1[x,0]+=e
				#self.B2+=e
				self.calculate_forward(X[i,:-1])
				perte1=-log(self.os[X[i,-1]-1,0])+perte1
				sumGrad+=self.S2[x,0]
			delta=(perte1-perte)/e

			if (sumGrad/delta > 1+seuil) or (sumGrad/delta < 1-seuil):
				wrong = True
				print  self.G2[x,0]/delta
			perte = 0.0
			perte1 = 0.0
			sumGrad = 0.0
		if not (wrong):
			print "Grandiant of B2 is checked"
		else:
			print "Grandiant of B2 is wrong"
		#print delta

	def verify_minBatch_W1(self,train_set,n):
		X=mat(train_set[:n,:])

		grad =  matrix(zeros((self.dh,self.d)))
		for i in range(n):
			self.calculate_forward(X[i,:-1])
			self.calculate_backward(X[i,:])
		   # W1=mat(dh*d)
			grad += self.aW1   # W2=mat(m*dh)
		return grad

	def verify_minBatch_W2(self,train_set,n):
		X=mat(train_set[:n,:])
		grad = matrix(zeros((self.m,self.dh)))
		for i in range(n):
			self.calculate_forward(X[i,:-1])
			self.calculate_backward(X[i,:])
		   # W1=mat(dh*d)
			grad += self.aW2   # W2=mat(m*dh)
		return grad

	def verify_minBatch_B1(self,train_set,n):
		X=mat(train_set[:n,:])
		grad = matrix(zeros((self.dh,1)))
		for i in range(n):
			self.calculate_forward(X[i,:-1])
			self.calculate_backward(X[i,:])
		   # W1=mat(dh*d)
			grad += self.S1   # W2=mat(m*dh)
		return grad

	def verify_minBatch_B2(self,train_set,n):
		X=mat(train_set[:n,:])
		grad =   matrix(zeros((self.m,1)))
		for i in range(n):
			self.calculate_forward(X[i,:-1])
			print X[i,:-1]
			print self.os
			self.calculate_backward(X[i,:])
		   # W1=mat(dh*d)
			grad += self.S2  # W2=mat(m*dh)
		return grad
