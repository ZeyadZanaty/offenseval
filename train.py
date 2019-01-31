from DataReader import DataReader
from Preprocessor import Preprocessor
from Vectorizer import Vectorizer
from Classifier import Classifier
from DeepLearning import DeepLearner
from sklearn.model_selection import train_test_split as split
import numpy as np

dr = DataReader('./datasets/training-v1/offenseval-training-v1.tsv','A')
data,labels = dr.get_labelled_data()
data,labels = dr.shuffle(data,labels,'random')

data = data[:]
labels = labels[:]

prp = Preprocessor('remove_stopwords','lemmatize')
data = prp.clean(data)

tr_data,tst_data,tr_labels,tst_labels = split(np.array(data),labels,test_size=0.2,stratify=labels)
tr_data,tr_labels = dr.upsample(tr_data,tr_labels,label=1)
tr_data,tr_labels = dr.shuffle(tr_data,tr_labels,'random')

vct = Vectorizer('count')
vct.vectorize(tr_data)

model=DeepLearner(tr_data,tr_labels,vocab_length=vct.vocab_length,model_type='CNN')
model.train(epochs=20)

acc = model.test_and_plot(tst_data,tst_labels)

print('Accuracy:',acc)