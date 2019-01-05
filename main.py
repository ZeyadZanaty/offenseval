from DataReader import DataReader
from Preprocessor import Preprocessor
from Vectorizer import Vectorizer
from Classifier import Classifier
from NeuralNetwork import NeuralNetwork
from sklearn.model_selection import train_test_split as split
import numpy as np

dr = DataReader('./datasets/training-v1/offenseval-training-v1.tsv','A')
data,labels = dr.get_labelled_data()
data,labels = dr.shuffle(data,labels,'random')

data = data[:]
labels = labels[:]

prp = Preprocessor('remove_stopwords','stem')
data = prp.clean(data)

vct = Vectorizer('tfidf')
vectors = vct.vectorize(data)
tr_vectors,tst_vectors,tr_labels,tst_labels = split(vectors,labels,test_size=0.2)

clf = Classifier('RandomForest',{'n_estimators':60})
tuned_clf = clf.tune(tr_vectors,tr_labels,{'n_estimators': [n for n in range(10,100,10)]},best_only=False)
print(tuned_clf)
print(clf.test_and_plot(tst_vectors,tst_labels,3))