from keras.datasets.mnist import load_data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.utils import to_categorical


def starter():
    (x_train, y_train) , (x_test, y_test) = load_data()

    np.random.seed(188)
    #vinay, kishre uncomment to see how the dataset looks like
    '''rand_mine = np.random.randint(0,x_train.shape[0],12)
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
    plt.close()
    
    ''' 

    image_height = x_train.shape[1]
    image_width = x_train.shape[2]
    num_channels = 1

    train_data = np.reshape(x_train, (x_train.shape[0], image_height, image_width, num_channels))
    test_data = np.reshape(x_test, (x_test.shape[0], image_height, image_width, num_channels))

    train_data = train_data.astype('float32')/255.
    test_data = test_data.astype('float32')/255.

    num_classes = 10
    train_labels_cat = to_categorical(y_train, num_classes)
    test_labels_cat = to_categorical(y_test, num_classes)

    print(train_labels_cat.shape) 
    print(test_labels_cat.shape)

    for _ in range(5):
        indexes = np.random.permutation(len(train_data))

    train_data = train_data[indexes]
    train_labels_cat = train_labels_cat[indexes]

    val_percent = 0.10
    val_count = int(val_percent*len(train_data))

    val_data = train_data[:val_count,:]
    val_labels_cat = train_labels_cat[:val_count,:]

    train_data_new = train_data[val_count:,:]
    train_labels_cat_new = train_labels_cat[val_count:,:]

    return train_data_new, train_labels_cat_new, val_data, val_labels_cat, test_data, test_labels_cat
