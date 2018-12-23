from DataReader import DataReader
from Preprocess import Preprocess

dr = DataReader('./datasets/training-v1/offenseval-training-v1.tsv')
data,labels = dr.get_labelled_data()
data = data[:50]
prp = Preprocess(data,labels)
prp_data = prp.clean(['remove_stopwords','lemmatize'])

for i in range(500):
    print('\n',data[i],labels[i])
    print('\t',prp_data[i])
