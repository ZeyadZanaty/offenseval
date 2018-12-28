import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np

class NeuralNetwork:

    def __init__(self,data,labels,vocab_length):
        self.tr_data, self.val_data, tr_labels, val_labels = train_test_split(np.array(data),labels, test_size=0.2, random_state=52)
        self.tr_labels = self.one_hot(tr_labels)
        self.val_labels = self.one_hot(val_labels)
        self.vocab_length = vocab_length
        self.build_model()
    
    def one_hot(self,labels):
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder()
        return encoder.fit_transform(np.array(labels).reshape(-1,1)).toarray()
    
    def build_model(self):
        # TODO:
        # build model
        self.model = model


    def train(self,epochs=100,batch_size=64):
        self.model.fit(self.tr_data, self.tr_labels, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(self.val_data, self.val_labels))  # starts training
