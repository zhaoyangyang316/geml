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
import tools

class NN:
    def softmax(self,X):
        # print X.shape
        #print X
        return exp(X)/sum(exp(X))

    def sigmoid(self,X):
        #print X
        return 1/(1+exp(-X))

    def __init__(self, m,d, alpha,lamda,batch,NN_type,Reg_type):
        self.best_model_valid_error = 99.9
        self.best_model_test_error = 99.9

        self.alpha = alpha
        self.m= m  #output dimension
        self.d=d   #input dimension
        self.batch=batch #longeur de batch
        self.lamda=lamda
        self.NN_type = NN_type
        self.Reg_type = Reg_type

        print "Initialize the NN model."

        if(NN_type == 1):
            self.W1 = matrix(zeros((1,self.d)))
            self.aW1 = matrix(zeros((1,self.d)))
            self.B1 = matrix(zeros((1,1)))
            self.GaW1 = matrix(zeros((1,self.d)))
            self.S1 = matrix(zeros((1,1)))
            self.G1 = matrix(zeros((1,1)))
            self.os = matrix(zeros((1,1)))
            self.oa = matrix(zeros((1,1)))
            print "The dimension of input is ",d
            print "The dimension of output is ",m

            #initialisation de W1
            interv=1/(d**0.5)

                #for j in range(self.d):
                #self.W1[0,j] = random.uniform(-interv,interv)

        else:
            self.W1 = matrix(zeros((self.m,self.d)))
            self.aW1 = matrix(zeros((self.m,self.d)))
            self.B1 = matrix(zeros((self.m,1)))
            self.GaW1 = matrix(zeros((self.m,self.d)))
            self.S1 = matrix(zeros((self.m,1)))
            self.G1 = matrix(zeros((self.m,1)))
            self.os = matrix(zeros((self.m,1)))

            print "The dimension of input is ",d
            print "The dimension of output is ",m

            #initialisation de W1
            #interv=1/(d**0.5)
            #for i in range(self.m):   #m*d
            #    for j in range(self.d):
            #       self.W1[i,j] = random.uniform(-interv,interv)

            print 'initialisation fini'

    def calculate_forward(self,X):   #calcule hs et os
        if (self.NN_type == 0):
            self.os = self.softmax(self.W1 * X.T + self.B1)
        elif(self.NN_type == 1):
            self.oa = self.W1 * X.T + self.B1
            #print self.oa
            self.os = self.sigmoid(self.oa)
            #print "os is ",self.os
            #print "oa is ",self.oa
        else:
            print "NN type unknown!"

    def calculate_forward_pred(self,X):   #calcule hs et os
        if (self.NN_type == 0):
            self.os = self.softmax(self.bestW1 * X.T + self.bestB1)
        elif(self.NN_type == 1):
            self.oa = self.bestW1 * X.T + self.bestB1
            self.os = self.sigmoid(self.oa)
            #print "os is ",self.os
            #print "oa is ",self.oa
        else:
            print "NN type unknown!"

#TODO here
    def calculate_backward(self,X):
        #sigma2 d(C)/d(oa)
        #print X
        #print self.NN_type
        if (self.NN_type == 0):
            for i in range(self.m):
                if (i == (X[0,-1]-1)):
                    #print self.os[i,0]
                    self.S1[i,0]= self.os[i,0] -1
                else:
                    self.S1[i,0]= self.os[i,0]

            self.aW1=self.S1 * X[0,:-1] 
        else:
            t = X[0,-1] -1
            #print "t is ",t

            #print self.os
            self.S1[0,0]= t/self.os[0,0] -(1-t)/(1-self.os[0,0])
            
            self.S1[0,0] = - self.S1[0,0] * (self.os[0,0]**2) * exp(-self.oa[0,0]) 

            #print "S1 is ",self.S1
            self.aW1=self.S1 * X[0,:-1]
            #print "aW1 is ",self.aW1
              

    def adjust_weight(self):
        #X est 1*d
        if (self.Reg_type == 1):
            self.W1 = self.W1 - self.alpha * self.GaW1 - self.lamda*self.W1
            self.B1 = self.B1 - self.alpha * self.G1 - self.lamda*self.B1
            #print "W1 is ",self.W1
            #print "B1 is ",self.B1
        else:
            self.W1 = self.W1 - self.alpha * self.GaW1 
            self.W1 = self.absClip(self.W1, self.lamda)

            self.B1 = self.B1 - self.alpha * self.G1
            self.B1 = self.absClip(self.B1, self.lamda)

        #print "W1 is ",self.W1
        #print "B1 is ",self.B1


    def absClip(self, X, minV):
        before = X.A
        absA = abs(before).clip(minV) - minV
        absA = sign(before) * absA
        return mat(absA)


    def train(self,train_set,valide_set,valide_labels,test_set,test_labels):
        nb_valide=valide_set.shape[0]
        changeTimer = 0
        print 'training rate is ',self.alpha
        finish = False
        nb_train=train_set.shape[0]
        max_iter = 10*nb_train
        itera=0
        seuil = 1
        taux = 100.0
        taux_valid =100.0
        k=0
        X=mat(train_set)
        print max_iter;
        while not finish:
            inds = range(train_set.shape[0])
            random.shuffle(inds)
            for i in range(self.batch):
                self.calculate_forward(X[inds[i],:-1])
                self.calculate_backward(X[inds[i],:])
                self.G1+=self.S1
                self.GaW1+=self.aW1
                itera+=1
                k+=1
                k=k%nb_train

            self.GaW1 = self.GaW1/self.batch
            self.G1 = self.G1/self.batch
            self.adjust_weight()

            if (itera%nb_train == 0):
                les_comptes=self.compute_predictions_train(test_set)
                classes_pred = argmax(les_comptes,axis=1)+1
                confmat = tools.teste(test_labels, classes_pred,self.m)
                sum_preds = sum(confmat)
                sum_correct = sum(diag(confmat))
                taux_test= 100*(1.0 - (float(sum_correct) / sum_preds))

                #taux de erreur pour validation
                les_comptes=self.compute_predictions_train(valide_set)
                classes_pred = argmax(les_comptes,axis=1)+1
                confmat = tools.teste(valide_labels, classes_pred,self.m)
                sum_preds = sum(confmat)
                sum_correct = sum(diag(confmat))
                taux_valid= 100*(1.0 - (float(sum_correct) / sum_preds))
                print "L'erreur de test est de validation est ", taux_valid,"%",time.clock()

                #taux de erreur pour train_set
                les_comptes=self.compute_predictions_train(train_set[:,:-1])
                classes_pred = argmax(les_comptes,axis=1)+1
                confmat = tools.teste(train_set[:,-1], classes_pred,self.m)
                sum_preds = sum(confmat)
                sum_correct = sum(diag(confmat))
                taux= 100*(1.0 - (float(sum_correct) / sum_preds))
                #print "iteration :", itera, ". L'erreur de test est de training est ", taux,"%",time.clock()
                #print "iteration :", itera
                print "L'erreur de test est de training est ", taux,"%",time.clock()


                if (taux_valid < self.best_model_valid_error):
                    self.best_model_valid_error = taux_valid
                    self.best_model_test_error = taux_test
                    print "Best model until now with error of validation:",self.best_model_valid_error,"%, computing time is ",time.clock()," seconds"
                    print "The best model has error rate ", self.best_model_test_error,"%, on test set. "
                    self.bestW1 = self.W1
                    self.bestB1 = self.B1

            self.G1 = self.G1*0.0
            self.GaW1 =self.GaW1*0.0



            if (taux < 30) and (changeTimer == 0):
                self.alpha=self.alpha/2
                changeTimer += 1
                print "alpha changed !!! because taux < 30"
                print self.alpha



            if (taux < 10) and (changeTimer == 1):
                self.alpha=self.alpha/2
                changeTimer += 1
                print "alpha changed !!! because taux < 10"
                print self.alpha


            if (taux_valid< seuil) or (itera > max_iter):
                self.bestW1 = self.W1
                self.bestB1 = self.B1
                finish = True



        print self.bestW1.shape
        print self.bestB1.shape


    def compute_predictions(self,testData):
        nb_test=testData.shape[0]
        sorties = mat(zeros((nb_test,self.m)))
        for i in range(nb_test):
            data=mat(testData[i,:]).T
            if (self.NN_type == 0):
                sorties[i,:]=(self.softmax(self.bestW1*data +self.bestB1)).T
            else:
                sorties[i,1]=(self.sigmoid(self.bestW1*data +self.bestB1)).T
                sorties[i,0]=1 -sorties[i,1]
        return sorties.A



    def compute_predictions_train(self,testData):
        nb_test=testData.shape[0]
        sorties = mat(zeros((nb_test,self.m)))
        for i in range(nb_test):
            data=mat(testData[i,:]).T
            if (self.NN_type == 0):
                sorties[i,:]=(self.softmax(self.W1*data +self.B1)).T
            else:
                sorties[i,1]=(self.sigmoid(self.W1*data +self.B1)).T     
                sorties[i,0]=1 -sorties[i,1]           
        return sorties.A
