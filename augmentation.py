import numpy as np
from skimage import transform
from skimage.transform import warp, AffineTransform
import tensorflow as tf
if tf.__version__ < "1.6.0" and os.environ.get('READTHEDOCS', None) != 'True':
        raise RuntimeError(
            "TensorLayer does not support Tensorflow version older than 1.6.0.\n"
            "Please update Tensorflow with:\n"
            " - `pip install --upgrade tensorflow`\n"
            " - `pip install --upgrade tensorflow-gpu`"
        )

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *



def rot_90_deg(X):
    X_aug = np.zeros_like(X)
    
    for i,img in enumerate(X):
        X_aug[i] = transform.rotate(img, 270)
    return X_aug

def rot_180_deg(X):
    X_aug = np.zeros_like(X)
    
    for i,img in enumerate(X):
        X_aug[i] = transform.rotate(img, 180)
    return X_aug

def rot_270_deg(X):
    X_aug = np.zeros_like(X)
    
    for i,img in enumerate(X):
        X_aug[i] = transform.rotate(img, 180)
    return X_aug

def pad_distort_im_fn(x):
    """ Zero pads an image to 40x40, and distort it.
    Examples
    ---------
    x = pad_distort_im_fn(X_train[0])
    print(x, x.shape, x.max())
    tl.vis.save_image(x, '_xd.png')
    tl.vis.save_image(X_train[0], '_x.png')
    """
    b = np.zeros((40, 40, 1))
    o = int((40-28)/2)
    b[o:o+28, o:o+28] = x
    x = b
    x = tl.prepro.rotation(x, rg=30, is_random=True, fill_mode='constant')
    x = tl.prepro.shear(x, 0.05, is_random=True, fill_mode='constant')
    x = tl.prepro.shift(x, wrg=0.25, hrg=0.25, is_random=True, fill_mode='constant')
    x = tl.prepro.zoom(x, zoom_range=(0.95, 1.05))
    return x

def pad_distort_ims_fn(X):
    """ Zero pads images to 40x40, and distort them. """
    X_40 = []
    for X_a, _ in tl.iterate.minibatches(X, X, 50, shuffle=False):
        X_40.extend(tl.prepro.threading_data(X_a, fn=pad_distort_im_fn))
    X_40 = np.asarray(X_40)
    return X_40

# create dataset with size of 40x40 with distortion
def aug_data(X_train,X_val,X_test):
    X_train_40 = pad_distort_ims_fn(X_train)
    X_val_40 = pad_distort_ims_fn(X_val)
    X_test_40 = pad_distort_ims_fn(X_test)
    return X_train_40,X_val_40,X_test_40
