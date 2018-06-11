from keras.models import Sequential
from keras.layers import Dense
import numpy
import keras
from keras import optimizers
import csv
from itertools import chain
from keras.models import load_model
from python_speech_features import mfcc
from python_speech_features import logfbank
import numpy
import scipy.io.wavfile as wav
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import train_test_split
from random import shuffle
from sklearn.externals import joblib

#dataset1 = csv.reader(open("testingfile.csv", "r"), delimiter=",")
#Data1 = list(dataset1)
#X1, Y1 = zip(*[(s[:-1], [s[-1]]) for s in Data1])
# split into input (X) and output (Y) variables

#x_test = numpy.array(X1).astype("float")
#x_test = x_test[800:900,:]

class conversation:

	def __init__(self):
		self.model = Sequential()
		self.model = load_model('my_model.h5')
		self.model.load_weights('model.h5')
 		#self.model.compile()
		self.model._make_predict_function()
		self.graph = tf.get_default_graph()


	def conversation_mode(self,rate,sig):
		#(rate,sig)=wav.read(sig)
		mfcc_feat = mfcc(sig,rate, winlen=0.025, winstep=0.01,nfft=512, numcep = 25)
		dt=DecisionTreeClassifier()
		#dt.fit(x,y)
		#joblib.dump(dt, 'tree.pkl')
		dt = joblib.load('tree.pkl')
		z=dt.predict(mfcc_feat)
		u=[0,0,0,0]
		for i in z:
		    if(i == 0):
		        u[0]=u[0]+1
		    elif(i == 1):
		        u[1]=u[1]+1
		    elif(i == 2):
		        u[2]=u[2]+1
		    elif(i == 3):
		        u[3]=u[3]+1

		for i in range(0,len(u)):
		     u[i]=u[i]/len(z)
		'''
		y_test=self.model.predict(x_test,batch_size=50)
		print(type(y_test))
		print(y_test.shape)
		count = [0,0,0,0,0]
		max_index = y_test.argmax(axis=1)
		for i in range(len(max_index)):
			count[max_index[i]]=count[max_index[i]]+1
		'''
		print(u)

		index=u.index(max(u))
		if(index==1):
			return('Aaditya')
		elif(index==0):
			return('Pranay')
		elif(index==3):
			return('Mridul')
		elif(index==2):
			return('Akshay')
		else:
			return('unknown')

#conversation_mode()

