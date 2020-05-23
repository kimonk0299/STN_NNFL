import numpy as np
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math

from samplerNinterpolation import sample_interpolate


def get_initial_weights(output_size):
	b = np.zeros((2, 3), dtype='float32')
	b[0, 0] = 1
	b[1, 1] = 1
	W = np.zeros((output_size, 6), dtype='float32')
	weights = [W, b.flatten()]
	return weights

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

def random_mini_batches(X, Y, mini_batch_size = 64):
	m = X.shape[0]
	mini_batches = []
	num_complete_minibatches = math.floor(m/mini_batch_size)

	for k in range(0, num_complete_minibatches):
		mini_batch_X = X[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :, :]
		mini_batch_Y = Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	if m % mini_batch_size != 0:
		mini_batch_X = X[num_complete_minibatches * mini_batch_size : m, :, :]
		mini_batch_Y = Y[num_complete_minibatches * mini_batch_size : m, :]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	return mini_batches

def append_shuffle(train_data,train_label,val_data,val_label,test_data,test_label,train_aug,val_aug,test_aug):
	m = np.size(train_data,0)
	b = np.zeros((m, 40, 40, 1))
	o = int((40-28)/2)
	b[:, o:o+28, o:o+28] = train_data
	train_data = np.squeeze(b)

	m = np.size(val_data,0)
	b = np.zeros((m, 40, 40, 1))
	o = int((40-28)/2)
	b[:, o:o+28, o:o+28] = val_data
	val_data = np.squeeze(b)

	m = np.size(test_data,0)
	b = np.zeros((m, 40, 40, 1))
	o = int((40-28)/2)
	b[:, o:o+28, o:o+28] = test_data
	test_data = np.squeeze(b)

	train_app = np.append(train_data,train_aug,axis=0)
	val_app = np.append(val_data,val_aug,axis=0)
	test_app = np.append(test_data,test_aug,axis=0)

	train_appl = np.append(train_label,train_label,axis=0)
	val_appl = np.append(val_label,val_label,axis=0)
	test_appl = np.append(test_label,test_label,axis=0)

	m = np.size(train_app,0)
	permutation = list(np.random.permutation(m))
	train_sh = train_app[permutation,:,:]
	train_sh_label = train_appl[permutation,:]

	m = np.size(val_app,0)
	permutation = list(np.random.permutation(m))
	val_sh = val_app[permutation,:,:]
	val_sh_label = val_appl[permutation,:]

	m = np.size(test_app,0)
	permutation = list(np.random.permutation(m))
	test_sh = test_app[permutation,:,:]
	test_sh_label = test_appl[permutation,:]

	return train_sh,train_sh_label,val_sh,val_sh_label,test_sh,test_sh_label
        
    
