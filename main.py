from DataReader import DataReader
from Preprocessor import Preprocessor
from Vectorizer import Vectorizer
from Classifier import Classifier
from DeepLearning import DeepLearner
from sklearn.model_selection import train_test_split as split
import numpy as np
lbls=['NOT','OFF']

dr_tr = DataReader('./datasets/training-v1/offenseval-training-v1.tsv','A')
tr_data,tr_labels = dr_tr.get_labelled_data()
tr_data,tr_labels = dr_tr.shuffle(tr_data,tr_labels,'random')

dr_tst = DataReader('./datasets/test-A/testset-taska.tsv')
tst_data,tst_ids = dr_tst.get_test_data()

tr_data = tr_data[:]
tr_labels = tr_labels[:]

prp = Preprocessor('remove_stopwords','stem')
tr_data = prp.clean(tr_data)
tst_data = prp.clean(tst_data)

vct = Vectorizer('tfidf')
tr_vectors = vct.vectorize(tr_data)
tst_vectors = vct.vectorize(tst_data)

clfs = Classifier('M-NaiveBayes')
tuned_clf = clf.tune(tr_vectors,tr_labels,{'alpha':[0,1,5,10],'fit_prior':[True,False]},best_only=False)
print(tuned_clf)

predictions = clf.predict(tst_vectors)
for i,id in enumerate(tst_ids):
    print(id,lbls[predictions[i]])