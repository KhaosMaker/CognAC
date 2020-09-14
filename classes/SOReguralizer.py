from tensorflow.keras import regularizers
from keras import backend as K
import numpy as np

class SOReguralizer(regularizers.Regularizer):
    def __init__(self, lamb):
        self.lamb = lamb

    def __call__(self, x):
        return self.cust_reg(x)
    
    def fro_norm(self, w):
        return K.sqrt(K.sum(K.square(K.abs(w))))

    def cust_reg(self, w):
        a = K.dot(K.transpose(w), w) - np.eye(w.shape[1])
        b = K.dot(w, K.transpose(w)) - np.eye(w.shape[0])
        res = self.lamb*(self.fro_norm(a) + self.fro_norm(b))
        return res
    
    def get_config(self):
        return {'lamb': self.lamb}     