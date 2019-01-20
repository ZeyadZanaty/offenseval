import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
from keras.layers import Input, Dense, Embedding, Convolution1D, MaxPooling1D, MaxPooling2D, Convolution2D, LSTM
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
        self.tr_data, self.val_data, tr_labels, val_labels = train_test_split(np.array(data),labels, test_size=0.35,stratify=labels)
        self.tr_labels = self.one_hot(tr_labels)
        self.val_labels = self.one_hot(val_labels)
        self.vocab_length = vocab_length
        self.max_len = max(len(max(self.tr_data,key=lambda x:len(x))),len(max(self.val_data,key=lambda x:len(x))))
        self.tr_data = self.encode_corpus(self.tr_data)
        self.val_data = self.encode_corpus(self.val_data)
        self.model_type = model_type
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
    
    def CNN_2D(self):
        model = Sequential()
        model.add(Embedding(self.vocab_length, 30, input_length=self.max_len))
        model.add(Reshape((30,self.max_len,1)))
        model.add(Convolution2D(32,(1,5),activation="relu"))
        model.add(Dropout(0.9))
        model.add(Convolution2D(16,(2,3),activation="relu"))
        model.add(Dropout(0.8))
        model.add(Convolution2D(16,(2,2),activation="relu"))
        model.add(Dropout(0.7))
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
    
    def visualize(self):
        from keras.utils import plot_model
        plot_model(self.model,show_layer_names=False,show_shapes=True, to_file='./docs/'+str(self.model_type)+'.png')
    
    def encode_corpus(self,data):
        encoded_docs = [one_hot(' '.join(d), self.vocab_length) for d in data]
        return pad_sequences(encoded_docs, maxlen=self.max_len, padding='post')

    def train(self,epochs=100,batch_size=64):
        self.model.fit(self.tr_data, self.tr_labels, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(self.val_data, self.val_labels))  # starts training

    def test(self,tst_data,tst_labels):
        tst_data = self.encode_corpus(tst_data)
        tst_labels = self.one_hot(tst_labels)
        return self.model.metrics_names,self.model.evaluate(tst_data, tst_labels, batch_size=64, verbose=1)
          
    def test_and_plot(self,tst_data,tst_labels,class_num=2):
        tst_data = self.encode_corpus(tst_data)
        tst_labels = self.one_hot(tst_labels)
        predicted_tst_labels = self.model.predict(tst_data,batch_size=64)
        conf = np.zeros([class_num,class_num])
        confnorm = np.zeros([class_num,class_num])
        for i in range(0,tst_data.shape[0]):
            j = np.argmax(tst_labels[i,:])
            k = np.argmax(predicted_tst_labels[i])
            conf[j,k] = conf[j,k] + 1
        for i in range(0,class_num):
            confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
        self._confusion_matrix(confnorm, labels=[i for i in range(class_num)])
        return self.model.metrics_names,self.model.evaluate(tst_data, tst_labels, batch_size=64, verbose=1)

    def _confusion_matrix(self,cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()     