import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
from keras.layers import Input, Dense, Embedding, Conv1D, MaxPooling1D, Convolution1D, LSTM
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.regularizers import l2, l1
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

class DeepLearner:

    def __init__(self,data,labels,vocab_length=0,model_type='LSTM'):
        self.tr_data, self.val_data, tr_labels, val_labels = train_test_split(np.array(data),labels, test_size=0.2, random_state=52)
        self.tr_labels = self.one_hot(tr_labels)
        self.val_labels = self.one_hot(val_labels)
        self.vocab_length = vocab_length
        self.max_len = max(len(max(self.tr_data,key=lambda x:len(x))),len(max(self.val_data,key=lambda x:len(x))))
        self.tr_data = self.encode_corpus(self.tr_data)
        self.val_data = self.encode_corpus(self.val_data)
        model_call = getattr(self,model_type,None)
        if model_call:
            model_call()
        else:
            raise Exception('No such model.')

    def one_hot(self,labels):
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder()
        return encoder.fit_transform(np.array(labels).reshape(-1,1)).toarray()
    
    def CNN(self):
        model = Sequential()
        model.add(Embedding(self.vocab_length, 30, input_length=self.max_len))
        model.add(Convolution1D(64,5,activation="relu"))
        model.add(Dropout(0.5))
        model.add(Convolution1D(32,3,activation="relu"))
        model.add(Dropout(0.5))
        model.add(Convolution1D(16,3,activation="sigmoid"))
        model.add(MaxPooling1D(5))
        model.add(Flatten())
        model.add(Dense(self.tr_labels.shape[1],activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        model.summary()
        self.model = model

    def LSTM(self):
        model = Sequential()
        model.add(Embedding(self.vocab_length, 30, input_length=self.max_len))
        model.add(LSTM(200))
        model.add(Dense(self.max_len, activation='relu', W_regularizer=l2(0.90)))
        model.add(Dense(self.tr_labels.shape[1], activation='softmax', W_regularizer=l2(0.1)))
        adam_1 = Adam(lr=0.008)
        model.compile(loss='categorical_crossentropy', optimizer=adam_1,metrics=['accuracy'])
        model.summary()
        self.model = model
    
    def encode_corpus(self,data):
        encoded_docs = [one_hot(' '.join(d), self.vocab_length) for d in data]
        return pad_sequences(encoded_docs, maxlen=self.max_len, padding='post')

    def train(self,epochs=100,batch_size=64):
        self.model.fit(self.tr_data, self.tr_labels, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(self.val_data, self.val_labels))  # starts training

    def test(self,tst_data,tst_labels):
        tst_data = self.encode_corpus(tst_data)
        tst_labels = self.one_hot(tst_labels)
        return self.model.metrics_names,self.model.evaluate(tst_data, tst_labels, batch_size=64, verbose=1)

