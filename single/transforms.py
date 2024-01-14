import numpy as np

def normalize(img):
    """Subtract mean, set STD to 1.0"""
    for i in range(img.shape[0]):
        img[i] -= np.mean(img[i])
        img[i] /= np.std(img[i])
    return img.astype(np.float32)

def normalize2(img):
    """Subtract mean, set STD to 1.0"""
    for i in range(img.shape[0]):
        img[i] /= max(15, np.percentile(img[i], 20))*5
        img[i] = np.arcsinh(img[i])
        img[i] -= np.mean(img[i])
        img[i] /= np.std(img[i])
    return img