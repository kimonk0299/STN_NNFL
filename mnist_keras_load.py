from keras.datasets.mnist import load_data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


(x_train, y_train) , (x_test, y_test) = load_data()

np.random.seed(188)

rand_mine = np.random.randint(0,x_train.shape[0],12)

sampled_x = x_train[rand_mine]
sampled_y = y_train[rand_mine]

num_rows = 2
num_cols = 6

f, ax = plt.subplots(num_rows, num_cols, figsize = (12,5), gridspec_kw = {'wspace':0.03 , 'hspace':0.01}, squeeze = True)

for i in range (num_rows):
    for j in range (num_cols):
        image_index = i*6 + j
        ax[i,j].axis("off")
        ax[i,j].imshow(sampled_x[image_index], cmap='gray')
        ax[i,j].set_title('No. %d' % sampled_y[image_index])

plt.show()
#plt.close()