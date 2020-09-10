from keras.layers import SimpleRNN, Masking, LSTM, Dense, GRU, TimeDistributed, Input
from keras.models import Sequential, Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K


from random import randint
from time import time
from classes.SOReguralizer import SOReguralizer

import numpy as np
#import keras
import pickle

"""
# FOR REPRODUCIBILITY
import numpy as np
# random seeds must be set before importing keras & tensorflow
my_seed = 112
np.random.seed(my_seed)
import random 
random.seed(my_seed)
import tensorflow as tf
tf.compat.v2.random.set_seed(my_seed)
tf.compat.v1.set_random_seed(my_seed)
"""




class Embedder():
    def __init__(self, level, unit1=1, unit2=1, special_c= -45000, lamb=0.01, orthogonal=False):
        self.level = level
        self.unit1 = unit1
        self.unit2 = unit2
        self.special_c = special_c
        self.lamb = lamb
        self.orthogonal = orthogonal
        self.reg = SOReguralizer(lamb)
        inputs = Input(shape=(None, 1), batch_shape=(1, None, 1), name='input')
        mask = Masking(mask_value=special_c, name='mask')(inputs)
        if orthogonal:
            rnn_layer_1, state_c1 = GRU(kernel_regularizer=self.reg, units=unit1, return_sequences=True, name='rnn_layer_1', return_state=True, recurrent_initializer="orthogonal", stateful=True)(mask)
            rnn_layer_2, state_c2 = GRU(kernel_regularizer=self.reg, units=unit2, return_sequences=True, name='rnn_layer_2', return_state=True, recurrent_initializer="orthogonal", stateful=True)(rnn_layer_1)
        else:
            rnn_layer_1, state_c1 = SimpleRNN(units=unit1, return_sequences=True, name='rnn_layer_1', return_state=True, recurrent_initializer="orthogonal", stateful=True)(mask)
            rnn_layer_2, state_c2 = SimpleRNN(units=unit2, return_sequences=True, name='rnn_layer_2', return_state=True, recurrent_initializer="orthogonal", stateful=True)(rnn_layer_1)
        
        output = TimeDistributed(Dense(1, kernel_initializer='normal',activation='linear', name='output'))(rnn_layer_2)
        self.model = Model(input=inputs, output=output)
        self.embedder = Model(inputs=self.model.input, outputs=[state_c1, state_c2])
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        self.emb_size = unit1+unit2

        self.earlyStopping = EarlyStopping(monitor='accuracy', patience=10, verbose=0, mode='min')
        #self.mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        self.reduce_lr_loss = ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

        self.partialState = None

    def fit(self, data, epochs=10, batch=1, specialValue = 40000):
        """
        data :- list of np array
        """
        tempModel = self.makeTempModel(batch)
        special_value = specialValue
        padded_seq = pad_sequences(data, padding='post', value=special_value)
        # batch size, timestamp, input dim

        X = np.reshape(padded_seq, (padded_seq.shape[0], padded_seq.shape[1], 1))
        Y = np.roll(X, 1, axis=1)

        tempModel.fit(x=X, y=Y, epochs=epochs, batch_size=batch, callbacks=[self.earlyStopping, self.reduce_lr_loss], verbose = 1)
        self.model.set_weights(tempModel.get_weights())
    
    def makeTempModel(self, batch):
        inputs = Input(shape=(None, 1), batch_shape=(None, None, 1), name='input')
        mask = Masking(mask_value=self.special_c, name='mask')(inputs)
        if self.orthogonal:
            rnn_layer_1, state_c1 = GRU(kernel_regularizer=self.reg, units=self.unit1, return_sequences=True, name='rnn_layer_1', return_state=True, recurrent_initializer="orthogonal")(mask)
            rnn_layer_2, state_c2 = GRU(kernel_regularizer=self.reg, units=self.unit2, return_sequences=True, name='rnn_layer_2', return_state=True, recurrent_initializer="orthogonal")(rnn_layer_1)
        else:
            rnn_layer_1, state_c1 = GRU(units=self.unit1, return_sequences=True, name='rnn_layer_1', return_state=True, recurrent_initializer="orthogonal")(mask)
            rnn_layer_2, state_c2 = GRU(units=self.unit2, return_sequences=True, name='rnn_layer_2', return_state=True, recurrent_initializer="orthogonal")(rnn_layer_1)
        
        output = TimeDistributed(Dense(1, kernel_initializer='normal',activation='linear', name='output'))(rnn_layer_2)
        model = Model(input=inputs, output=output)
        model.set_weights(self.model.get_weights())
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        return model


    def get_embedding(self, seq):
        """
        seq = nparray with the chunk
        return :- nparray (1, self.emb_size)
        """        
        # batch size, timestamp, input dim

        if self.partialState is None:
            x = np.reshape(seq, (1, seq.shape[0], 1))       
            embedding = self.embedder.predict(x)
        else:            
            x = np.reshape(seq[-1], (1, 1, 1))
            self.loadPartialState()
            embedding = self.embedder.predict(x)             
        self.savePartialState(embedding)      
        res = np.concatenate(embedding, axis=1)[0]
        return res

    def getBatchEmbedding(self, seq):
        x = np.reshape(seq, (seq.shape[0],  seq.shape[1], 1))
        embedding = self.embedder.predict(x, batch_size=seq.shape[0], verbose=1)
        embedding = np.array(embedding)
        res = np.concatenate(embedding, axis=1)
        return res

    # SAVE MODEL
    def save(self, directory=""):
        """
        self.model.save(directory+'/emb_model_lv_'+str(self.level)+'.h5')
        self.embedder.save(directory+'/embedder'+str(self.level)+'.h5')
        with open(directory+"/emb_size.b", "wb") as f:
            pickle.dump(self.emb_size, f)
        """
        self.model.save_weights(directory+'/emb_model_lv_'+str(self.level)+'.h5')


    
    def load(self, directory="", level=None):
        if level is None:
            raise "Error: level parameter cannot be None!"
        self.level = level
        self.model.load_weights(directory+'/emb_model_lv_'+str(self.level)+'.h5')
        #self.model = load_model(directory+'/emb_model_lv_'+str(level)+'.h5', custom_objects={'reg':SOReguralizer})
        #self.embedder = load_model(directory+'/embedder'+str(level)+'.h5')
        #with open(directory+"/emb_size.b", "rb") as f:
        #    self.emb_size = pickle.load(f)


    def savePartialState(self, states):
        self.partialState = [states[0], states[1]]
    
    def loadPartialState(self):
        self.model.layers[2].reset_states(states=[self.partialState[0]])
        self.model.layers[3].reset_states(states=[self.partialState[1]])

    
    def resetPartialState(self):
        self.model.layers[2].reset_states()
        self.model.layers[3].reset_states()
        self.partialState = None


