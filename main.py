import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from mnist_keras_load import starter
from augmentation import aug_data
from appendNshuffle import append_shuffle
import numpy as np
from utils import get_initial_weights,STN,random_mini_batches

def main():
	train_data, train_label, val_data, val_label, test_data, test_label = starter()
	train_aug,val_aug,test_aug = aug_data(train_data,val_data,test_data)
	train_sh,train_sh_label,val_sh,val_sh_label,test_sh,test_sh_label = append_shuffle(train_data,train_label,val_data,val_label,test_data,test_label,train_aug,val_aug,test_aug)
	'''rand_mine = np.random.randint(0,val_sh.shape[0],12)
	sampled_x = val_sh[rand_mine]
	sampled_y = val_sh_label[rand_mine].reshape(12,10)
	num_rows = 2
	num_cols = 6
	f, ax = plt.subplots(num_rows, num_cols, figsize = (12,5), gridspec_kw = {'wspace':0.03 , 'hspace':0.01}, squeeze = True)
	for i in range (num_rows):
		for j in range (num_cols):
			image_index = i*6 + j
			ax[i,j].axis("off")
			ax[i,j].imshow(np.squeeze(sampled_x[image_index]), cmap='gray')
			ax[i,j].set_title('No. %d' % np.where(sampled_y[image_index] == 1))
	plt.show()'''
	minibatch_size = 256
	num_epochs = 100
	
	model = STN()
	model.compile(loss='categorical_crossentropy', optimizer='adam', learning_rate = 1e-3)
	print("network summary")
	model.summary()
	
	'''m = train_sh.shape[0]
	
        for epoch in range(num_epochs):
		num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
		minibatches = random_mini_batches(train_sh, train_sh_label, minibatch_size)
		
		for minibatch in minibatches:
			(minibatch_x,minibatch_y) = minibatch
			loss = model.train_on_batch(minibatch_x, minibatch_y)
			if epoch_arg % 10 == 0:
				val_score = model.evaluate(*val_data, verbose=1)
				test_score = model.evaluate(*test_data, verbose=1)
				message = 'Epoch: {0} | Val: {1} | Test: {2}'
				print(message.format(epoch, val_score, test_score))'''
				
			
			

	
	
	

	
	
	
	
	
