import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
from keras.layers import Input, Dense, Embedding, Conv1D, MaxPooling1D, Convolution1D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np

class NeuralNetwork:

    def __init__(self,data,labels,vocab_length=0):
        self.tr_data, self.val_data, tr_labels, val_labels = train_test_split(np.array(data),labels, test_size=0.5, random_state=52)
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
        model = Sequential()
        model.add(Reshape((self.tr_data.shape[1],1), input_shape=(self.tr_data.shape[1],)))
        model.add(Dropout(0.5))
        model.add(Convolution1D(32,50,activation="relu"))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.5))
        model.add(Convolution1D(32,50,activation="relu"))
        model.add(Dropout(0.5))
        model.add(Convolution1D(16,50,activation="sigmoid"))
        model.add(MaxPooling1D(3))
        model.add(Flatten())
        model.add(Dense(self.tr_labels.shape[1],activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        model.summary()
        self.model = model


    def train(self,epochs=200,batch_size=64):
        self.model.fit(self.tr_data, self.tr_labels, epochs=epochs, verbose=1, validation_data=(self.val_data, self.val_labels))  # starts training

    def test(self,tst_data,tst_labels):
        tst_data = np.array(tst_data)
        tst_labels = self.one_hot(tst_labels)
        return self.model.evaluate(tst_data, tst_labels, batch_size=64, verbose=1)



