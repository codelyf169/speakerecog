from keras.models import Sequential,load_model
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import train_test_split
from random import shuffle
from sklearn.externals import joblib
import keras
from keras import optimizers
import csv
from itertools import chain
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from random import shuffle
from keras.layers import Dense, Dropout, Activation
import tensorflow as tf
from pydub import AudioSegment

# fix random seed for reproducibility
np.random.seed(10)

class recognition:
	def __init__(self,username):
		#print (username)
		#self.model = Sequential()
		#username_model = username + '.h5'
		#username_weights = username + '_weights.h5'
		#self.model = load_model(username_model)
		#self.model.load_weights(username_weights)
 		#self.model.compile()
		#self.model._make_predict_function()
		#graph = tf.get_default_graph()
		self.dt=DecisionTreeClassifier()
		self.dt = joblib.load(username+'.pkl')
		


	def recog(self,mfcc_feat):
		x_test = mfcc_feat
		z=self.dt.predict(x_test)
		u=[0,0]
		for i in z:
		    if(i == 0):
		        u[0]=u[0]+1
		    elif(i == 1):
		        u[1]=u[1]+1
		for i in range(0,len(u)):
			u[i]=u[i]/len(z)
		#if(y_test > 0.8):
		#	return y_test
		return u[0]


class registration:
	def encode_target(self,df, target_column):
		    df_mod = df.copy()
		    #print(df_mod.shape)
		    targets = df_mod[target_column].unique()
		    map_to_int = {name: n for n, name in enumerate(targets)}
		    #print(map_to_int)
		    df_mod["Target"] = df_mod[target_column].map(map_to_int)
		    return (df_mod, targets)

	def reg_train(self,filename,username):

		file = open("temp.csv","w")
		lst=[]
		for i in range(0,25):
			lst.append('feature_'+str(i))
		lst.append("Result")
		str1=','.join(str(e) for e in lst)
		str1=str1+'\n'
		file.write(str1)
		(rate,sig)=wav.read(filename)
		mfcc_feat = mfcc(sig,rate, winlen=0.025, winstep=0.01,nfft=512, numcep = 25)
		for i in range(len(mfcc_feat)):
			lst = list(mfcc_feat[i])
			lst.append('True')
			str1=','.join(str(e) for e in lst)
			str1=str1+'\n'
			file.write(str1)
		(rate,sig) = wav.read("noise.wav")
		mfcc_feat = mfcc(sig,rate, winlen=0.025, winstep=0.01,nfft=512, numcep=25 )
		for i in range(len(mfcc_feat)):
			lst = list(mfcc_feat[i])
			lst.append('False')
			str1=','.join(str(e) for e in lst)
			str1=str1+'\n'
			file.write(str1)
		file.close()

		df=pd.read_csv("temp.csv")
		df = df.sample(frac=1).reset_index(drop=True)
		#print(df.shape)




		df1,targets = self.encode_target(df,"Result")
		#print(targets)
		print(df1[["Target", "Result"]].head())
		#test_idx=[120,449,1040]

		#print(df1[["Target", "Result"]])
		#print(df1[["Target", "Result"]].tail())

		del(df1["Result"])

		features = list(df1.columns[:25])
		#print(features)


		#print(df2.shape)

		#myarray = np.asarray(test_data)
		#print(myarray.shape)
		y=df1["Target"]
		x=df1[features]
		#print(x.shape)
		
		#df2=pd.read_csv("aaditya_test_2.csv")
		#X= np.asarray(df2)
		dt=DecisionTreeClassifier()
		dt.fit(x,y)
		joblib.dump(dt, username+'.pkl')
		'''
		dataset = csv.reader(open("temp.csv", "r"), delimiter=",")
		Data = list(dataset)
		dataset1 = csv.reader(open("permanent.csv", "r"), delimiter=",")
		Data1 = list(dataset1)
		Data2 = Data + Data1
		shuffle(Data2)
		X, Y = zip(*[(s[:-1], [s[-1]]) for s in Data2])
		# split into input (X) and output (Y) variables

		x_train = numpy.array(X).astype("float")
		y_train = numpy.array(Y).astype("float")

		#print("before change {} {}".format(x_train.shape,y_train.shape))
		x_test = x_train[15000:,:]
		y_test = y_train[15000:,:]
		#print("after change")

		x_train = x_train[0:15000,:]
		y_train = y_train[0:15000,:]
		#print("after second change")

		#print("xtrain: {}".format(x_train[6:10,17:22]))
		#print("ytrain: {}".format(y_train[6:10]))

		model = Sequential()
		model.add(Dense(40, input_dim=25, init='normal', activation='relu'))
		model.add(Dense(50, init='normal', activation='relu'))
		model.add(Dropout(0.4))
		model.add(Dense(50, init='normal', activation='relu'))
		model.add(Dropout(0.4))
		model.add(Dense(50, init='normal', activation='relu'))
		model.add(Dense(50, init='normal', activation='relu'))
		model.add(Dropout(0.4))
		model.add(Dense(50, init='normal', activation='relu'))
		model.add(Dropout(0.4))
		model.add(Dense(50, init='normal', activation='relu'))
		model.add(Dense(20, init='normal', activation='relu'))
		model.add(Dense(1, init='normal', activation='sigmoid'))
		#print ('Done...')
		sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='binary_crossentropy', optimizer='sgd' , metrics=['accuracy'])

		model.fit(x_train, y_train, epochs=4, batch_size=32)
		metric = model.evaluate(x_train,y_train)
		print("train accuracy: {}".format(metric[1]*100))

		username_model = username + '.h5'
		username_weights = username + '_weights.h5'

		model.save(username_model, overwrite=True)
		model.save_weights(username_weights, overwrite=True)
		# evaluate the model
		scores = model.evaluate(x_test, y_test)
		print("\ntest %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
		''' 

#obj = registration()
#obj.reg_train('akshay.wav','akshay')

