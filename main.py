def main():
	'''train_data, train_label, val_data, val_label, test_data, test_label = starter()
	train_aug,val_aug,test_aug = aug_data(train_data,val_data,test_data)
	train_sh,train_sh_label,val_sh,val_sh_label,test_sh,test_sh_label = append_shuffle(train_data,train_label,val_data,val_label,test_data,test_label,train_aug,val_aug,test_aug)
	rand_mine = np.random.randint(0,val_sh.shape[0],12)
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
