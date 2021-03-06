import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
from samplerNinterpolation import sample_interpolate
from utils import get_initial_weights


def STN_Model(input_shape=(40, 40, 1), sampling_size=(40, 40), num_classes=10):
	#Input
	STN_Input = keras.Input(shape=input_shape, name = 'STN_Input')

	#Layers for localization network
	locnet = layers.Conv2D(16, (3,3), activation = 'relu')(STN_Input)
	locnet = layers.MaxPool2D(pool_size=(2, 2))(locnet)
	locnet = layers.Conv2D(8, (4,4), activation = 'relu')(locnet)
	locnet = layers.MaxPool2D(pool_size=(2, 2))(locnet)
	locnet = layers.Conv2D(20, (5, 5), activation = 'relu')(locnet)
	locnet = layers.Flatten()(locnet)
	locnet = layers.Dense(50)(locnet)
	locnet = layers.Activation('relu')(locnet)
	weights = get_initial_weights(50)
	locnet = layers.Dense(6, weights=weights)(locnet)

	# Grid generator and bilenear interpolator layer
	sampler = sample_interpolate(sampling_size)([STN_Input, locnet])

	# Classification layer
	classifier = layers.Conv2D(32, (3, 3), padding='same', activation = 'relu')(sampler)
	classifier = layers.MaxPool2D(pool_size=(2, 2))(classifier)
	classifier = layers.Conv2D(16, (3, 3), activation = 'relu')(classifier)
	classifier = layers.MaxPool2D(pool_size=(2, 2))(classifier)
	classifier = layers.Flatten()(classifier)
	classifier = layers.Dense(256)(classifier)
	classifier = layers.Activation('relu')(classifier)
	classifier = layers.Dense(num_classes)(classifier)
	classifier_output = layers.Activation('softmax')(classifier)

	model = keras.Model(inputs=STN_Input, outputs=classifier_output)
	
	return model
