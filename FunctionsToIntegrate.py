from python_speech_features import fbank, delta, mfcc
import numpy as np
import librosa
import Constants as CNST
from keras.models import load_model
from model import speaker_representation
import tensorflow as tf
import scipy.io.wavfile as wav


class model_holder:
	def __init__(self):
		self.model = speaker_representation(CNST.INPUT_SHAPE)
		#self.model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
		self.model.load_weights("small_siamese_weights.h5",by_name=True)
		self.model._make_predict_function()
		self.graph = tf.get_default_graph()

	#function to be used while registering a new user
	#pass 30 second wav file path to this function
	#returns a numpy array (shape = (512,)) of 512 values to be stored in database as the speaker representation
	def generate_representation(self,voicefilepath=None):
		#audio, sr = librosa.load(voicefilepath, sr=CNST.SAMPLE_RATE, mono=False)
		sr,audio = wav.read(voicefilepath)
		#audio = audio.flatten()
		if(audio.shape[0]<CNST.SAMPLE_RATE*5*6):
			print("provide input file of 30 sec or more")
			return
		
		audios = []
		features = []
		for i in range(6):
			audios.append(audio[i*5*CNST.SAMPLE_RATE:CNST.SAMPLE_RATE*(i+1)*5,:])
			f1 = mfcc(audios[i], samplerate=CNST.SAMPLE_RATE, numcep=25)
			#f2 = delta(f1,2)
			#f3 = delta(f2,2)
			#feature = [f1,f2,f3]
			#feature = np.array(feature)
			#feature = np.rollaxis(feature,2,0)
			#feature = np.rollaxis(feature,2,0)
			features.append(f1)


		representation = np.zeros(CNST.REPR_SIZE)
		for i in range(6):
			input = [features[i]]
			input = np.array(input)
			prediction = self.model.predict(input)
			representation += prediction[0]

		representation /= 6
		return representation


	#function to be used to compare a speaker representation with a 5 sec voice sample wav file
	#returns a similarity score between the two
	#pass an array i.e. list of values of the representation and file path of the new 5 sec audio file
	#return value between [-1,1]. -1 indicates worst match, 1 indicates best match.
	def similarity_score(self,representation=None,voicefilepath=None):

		representation = np.array(representation)
		#audio, sr = librosa.load(voicefilepath, sr=CNST.SAMPLE_RATE, mono=False)
		#audio = audio.flatten()
		sr,audio = wav.read(voicefilepath)
		if(audio.shape[0]<CNST.SAMPLE_RATE*5):
			print("provide input file of 5 sec or more")
			return

		audio = audio[0:CNST.SAMPLE_RATE*5,:]

		repr_to_check = self.generate_representation_array(audio)

		similarity = np.dot(repr_to_check,representation)
		return similarity

	def generate_representation_array(self,nparray=None):

		if nparray is not None:
			#nparray = np.reshape(nparray,(nparray.shape[0],))
			if(nparray.shape[0]<CNST.SAMPLE_RATE*5):
				print("provide input file of 5 sec or more")
				return

			nparray = nparray[0:CNST.SAMPLE_RATE*5,:]
			f1 = mfcc(nparray, samplerate=CNST.SAMPLE_RATE, numcep=25)
			#f2 = delta(f1,2)
			#f3 = delta(f2,2)
			#feature = [f1,f2,f3]
			#feature = np.array(feature)
			#feature = np.rollaxis(feature,2,0)
			#feature = np.rollaxis(feature,2,0)
			input = [f1]
			input = np.array(input)
			with self.graph.as_default():
			    repr = self.model.predict(input)
			repr = repr[0]
			return repr

		else:
			print("nparray is empty")
