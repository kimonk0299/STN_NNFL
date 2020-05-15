import numpy as np
from skimage import transform
from skimage.transform import warp, AffineTransform

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

