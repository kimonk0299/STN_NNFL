import numpy as np
from skimage import transform
import tensorflow as tf

def append(train_data,train_label,val_data,val_label,test_data,test_label,train_aug,val_aug,test_aug):
	m = np.size(train_data,0)
	b = np.zeros((m, 40, 40, 1))
	o = int((40-28)/2)
	b[m, o:o+28, o:o+28, 1] = train_data
	train_data = b
	
	m = np.size(val_data,0)
	b = np.zeros((m, 40, 40, 1))
	o = int((40-28)/2)
	b[m, o:o+28, o:o+28, 1] = val_data
	val_data = b
	
	m = np.size(test_data,0)
	b = np.zeros((m, 40, 40, 1))
	o = int((40-28)/2)
	b[m, o:o+28, o:o+28, 1] = test_data
	test_data = b
	
	train_app = np.append(train_data,train_aug,axis=0)
	val_app = np.append(val_data,val_aug,axis=0)
	test_app = np.append(test_data,test_aug,axis=0)
	
	train_appl = np.append(train_label,train_label,axis=0)
	val_appl = np.append(val_label,val_label,axis=0)
	test_appl = np.append(test_label,test_label,axis=0)
	
	return train_app,train_appl,val_app,val_appl,test_app,test_appl
  
