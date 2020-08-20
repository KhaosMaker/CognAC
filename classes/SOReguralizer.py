from tensorflow.keras import regularizers
from keras import backend as K
import numpy as np

class SOReguralizer(regularizers.Regularizer):
    def __init__(self, lamb):
        self.lamb = lamb

    def __call__(self, x):
        return self.cust_reg(x)

    
    def fro_norm(self, w):
        return self.lamb*K.sqrt(K.sum(K.square(K.abs(w))))

    def cust_reg(self, w):
        m = K.dot(K.transpose(w), w) - np.eye(w.shape[1])
        return self.fro_norm(m)
    
    def get_config(self):
        return {'lamb': self.lamb}
        