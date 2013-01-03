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
import numpy
import random
import pylab
import time
import NN
import NN_m
import tools



import gzip,pickle

def run_classify(dh,alpha,lamda,batch):
	

	f=gzip.open('mnist.pkl.gz')
	mnist=pickle.load(f)

	n_exemple=mnist[0][0].shape[0]
	d=mnist[0][0].shape[1]

	data=numpy.zeros((n_exemple,d+1))
	data[:,:-1]=mnist[0][0]# matrice de train data
	data[:,:-1]=utilitaires.normalisation(data[:,:-1])
	aux=mnist[0][1]# vecteur des train labels
	print numpy.max(aux)

	for i in range(n_exemple):
		data[i,-1]=aux[i]+1
	print data.shape
	test_data=mnist[2][0]# data des valid labels
	test_data=utilitaires.normalisation(test_data)
	test_labels_full=mnist[2][1]+1# vecteur des valid labels
	print numpy.min(test_labels_full)

	#data[2][0]# matrice de test data
	#data[2][0]# vecteur des test labels

	valide_set_full=mnist[1][0]
	valide_set_full=utilitaires.normalisation(valide_set_full)
	valide_labels_full=mnist[1][1]+1
	print data.shape


	# Nombre de voisins (k) dans k-PPV
	#k = 2
	# Nombre de classes
	n_classes = 10
	# Nombre de points d'entrainement
	n_train = 40000
	n_test =1000
	n_valide=800

	print "On va entrainer"

	# decommenter pour avoir des resultats non-deterministes 
	random.seed(75595)
	# Déterminer au hasard des indices pour les exemples d'entrainement et de test
	inds_train = range(data.shape[0])
	inds_test = range(test_data.shape[0])
	inds_valide = range(valide_set_full.shape[0])

	random.shuffle(inds_train)
	random.shuffle(inds_test)
	random.shuffle(inds_valide)
	train_inds = inds_train[:n_train]
	test_inds = inds_test[:n_test]
	valide_inds = inds_test[:n_test]
	# separer les donnees dans les deux ensembles
	train_set = data[train_inds,:]
	test_set = test_data[test_inds,:]
	test_labels= test_labels_full[test_inds,:]
	valide_set = valide_set_full[valide_inds,:]
	valide_labels= valide_labels_full[valide_inds,:]

	print numpy.max(test_labels)
	# separarer l'ensemble de test dans les entrees et les etiquettes
	# Créer le classifieur
	print test_set.shape
	m=10
	#dh=200
	#alpha=0.001
	#lamda=0.1
	#batch=40
	model = NN_m.NN_m( m,d,dh, alpha,lamda,batch)
	#model.testGrad(train_set)
	# l'entrainer
	model.train(train_set,valide_set,valide_labels,test_set,test_labels)
	# Obtenir ses prédictions
	t1 = time.clock()
	les_comptes = model.compute_predictions(test_set)
	t2 = time.clock()
	print 'Ca nous a pris ', t2-t1, ' secondes pour calculer les predictions sur ', test_set.shape[0],' points de test'
	print les_comptes
	# Vote majoritaire (+1 puisquie nos classes sont de 1 a n)
	classes_pred = numpy.argmax(les_comptes,axis=1)+1
	print classes_pred.shape
	print test_labels.shape
	#print sum((classes_pred-test_labels)**2)/200
	# Faire les tests
	# Matrice de confusion 
	confmat = utilitaires.teste(test_labels, classes_pred,n_classes)
	print 'La matrice de confusion est:'
	print confmat

	# Erreur de test
	sum_preds = numpy.sum(confmat)
	sum_correct = numpy.sum(numpy.diag(confmat))
	print "L'erreur de test est de ", 100*(1.0 - (float(sum_correct) / sum_preds)),"%"


cal(500,0.01,0.01,20)