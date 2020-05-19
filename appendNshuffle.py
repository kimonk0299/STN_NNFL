import numpy as np
from skimage import transform
import tensorflow as tf

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
  
