import numpy as np
import numpy.random as rng
import random
import matplotlib.pyplot as plt

from keras.layers import Input, Lambda, Conv2D, Merge, Dense, MaxPooling2D, Dropout
from keras.layers import Dot, merge, Activation, Flatten, Add, Activation, Reshape, BatchNormalization
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
from keras.layers.core import Lambda
import keras.backend as K
import keras.utils as ku
import keras
import tensorflow as tf

import Constants as CNST

def similarity(tensor):
	x,y = tensor
	return tf.matmul(x,tf.transpose(y))

def scale(tensor):
	return tf.divide(tf.add(tensor,1),2)

def speaker_representation(input_shape=None):

	model = Sequential()
	'''
	model.add(Conv2D(64,kernel_size=(5,5),strides=(1,1),input_shape=input_shape,name='conv1',activation='relu'))
	#model.add(Dropout(0.2))
	model.add(BatchNormalization(name='bnorm1'))
	model.add(Conv2D(128,kernel_size=(5,5),strides=(2,2),name='conv2',activation='relu'))
	model.add(BatchNormalization(name='bnorm2'))
	'''
	model.add(Dense(100,name='dense1',input_shape=input_shape))
	model.add(Flatten())
	model.add(Dense(512,name='dense2'))
	model.add(BatchNormalization(name='bnorm1'))
	model.add(Dense(CNST.REPR_SIZE,name='repr'))
	model.add(Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln'))
	return model

def siamese_model(input_shape=None):
	input_1 = Input(shape=input_shape)
	input_2 = Input(shape=input_shape)

	repr_model = speaker_representation(input_shape)
	speaker_1 = repr_model(input_1)
	speaker_2 = repr_model(input_2)
	#print(speaker_1.get_shape())

	distance = Lambda(similarity)([speaker_1,speaker_2])
	#print(distance.get_shape())

	scaled_distance = Lambda(scale)(distance)
	s_model = Model(inputs=[input_1,input_2],outputs=scaled_distance)
	return s_model

if __name__=="__main__":
	md = siamese_model(CNST.INPUT_SHAPE)